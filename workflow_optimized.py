from langgraph.graph import StateGraph,START,END
from pydantic import BaseModel,Field
from typing import Annotated,List,TypedDict,Literal,Optional
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage,SystemMessage
from langgraph.types import Send
from langchain_community.tools.tavily_search import TavilySearchResults
import operator
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv
load_dotenv()
from pathlib import Path
import os
import re





class routerdecision(BaseModel):
    need_research:bool
    mode:Literal['closed_book','hybrid','open_book']
    queries:List[str]=Field(default_factory=list)

class evidenceitem(BaseModel):
    title: str
    url: str
    published_at: Optional[str] = None  
    snippet: Optional[str] = None
    source: Optional[str] = None

class evidencepack(BaseModel):
    evidence:List[evidenceitem]=Field(default_factory=list)


class TokenUsage(BaseModel):
    """Token usage reported by one LLM call."""
    agent_name: str
    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)
    total_tokens: int = Field(default=0, ge=0)


def _token_usage(response, agent_name: str) -> dict:
    usage = getattr(response, "usage_metadata", None) or {}
    input_tokens = usage.get("input_tokens", 0)
    output_tokens = usage.get("output_tokens", 0)
    total_tokens = usage.get("total_tokens", input_tokens + output_tokens)
    return TokenUsage(
        agent_name=agent_name,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=total_tokens,
    ).model_dump()


_STOP_WORDS = {
    "the", "and", "for", "with", "from", "that", "this", "into", "what",
    "when", "where", "how", "why", "are", "was", "were", "will", "can",
    "could", "should", "about", "their", "there", "have", "has", "using",
    "use", "your", "you", "its", "also", "than", "then", "such", "only",
}

class Task(BaseModel):
    id: int
    title: str

    goal: str = Field(
        ...,
        description="One sentence describing what the reader should be able to do/understand after this section.",
    )
    bullets: List[str] = Field(
        ...,
        min_length=3,
        max_length=6,
        description="3–6 concrete, non-overlapping subpoints to cover in this section.",
    )
    target_words: int = Field(..., description="Target word count for this section (120–550).")

    tags: List[str] = Field(default_factory=list)
    requires_research: bool = False
    requires_citations: bool = False
    requires_code: bool = False

def _select_task_evidence(task: Task, evidence: list[evidenceitem], limit: int = 4):
    """Choose a small, task-relevant evidence subset without another LLM call."""
    terms = set(
        re.findall(r"[a-z0-9]{3,}", " ".join(
            [task.title, task.goal, *task.bullets, *task.tags]
        ).lower())
    ) - _STOP_WORDS

    scored = []
    for item in evidence:
        text = f"{item.title} {item.snippet or ''}".lower()
        item_terms = set(re.findall(r"[a-z0-9]{3,}", text)) - _STOP_WORDS
        score = len(terms & item_terms)
        if score:
            scored.append((score, item))

    scored.sort(key=lambda pair: pair[0], reverse=True)
    selected = [item for _, item in scored[:limit]]

    # Fall back to a few sources if lexical matching finds nothing.
    return selected or evidence[:min(limit, len(evidence))]




class Plan(BaseModel):
    blog_title:str
    audience:str=Field(...,description="who this blog is for")
    tone:str=Field(...,description="writing tone(e.g.,practical,crisp)")
    blog_kind: Literal["explainer", "tutorial", "news_roundup", "comparison", "system_design"] = "explainer"
    constraints: List[str] = Field(default_factory=list)
    tasks:List[Task]


class State(TypedDict):
    topic:str

    mode: str
    need_research: bool
    queries: List[str]
    evidence: List[evidenceitem]
    plan: Optional[Plan]

    #reducer:result from workers get concate autometically
    sections: Annotated[List[tuple[int, str]], operator.add]
    token_usage: Annotated[List[dict], operator.add]
    merged_md: str
    md_with_placeholders: str
    image_specs: List[dict]
    final:str

class ImageSpec(BaseModel):
    placeholder: str = Field(..., description="e.g. [[IMAGE_1]]")
    filename: str = Field(..., description="Save under images/, e.g. qkv_flow.png")
    alt: str
    caption: str
    prompt: str = Field(..., description="Prompt to send to the image model.")
    


class GlobalImagePlan(BaseModel):
    images: List[ImageSpec] = Field(default_factory=list)



model = ChatGroq(
    model="openai/gpt-oss-120b",
    api_key=os.environ.get("GROQ_API_KEY"),
    max_retries=6,
)



ROUTER_SYSTEM = """You are a routing module for a technical blog planner.

Decide whether web research is needed BEFORE planning.

Modes:
- closed_book (needs_research=false):
  Evergreen topics where correctness does not depend on recent facts (concepts, fundamentals).
- hybrid (needs_research=true):
  Mostly evergreen but needs up-to-date examples/tools/models to be useful.
- open_book (needs_research=true):
  Mostly volatile: weekly roundups, "this week", "latest", rankings, pricing, policy/regulation.

If needs_research=true:
- Output 2–4 high-signal queries.
- Queries should be scoped and specific (avoid generic queries like just "AI" or "LLM").
- If user asked for "last week/this week/latest", reflect that constraint IN THE QUERIES.
"""

def router_node(state:State):
    topic=state['topic']
    decider = model.with_structured_output(routerdecision, include_raw=True)
    result = decider.invoke([
        SystemMessage(content=ROUTER_SYSTEM),
        HumanMessage(content=f"topic: {topic}"),
    ])
    decision = result["parsed"]
    return {
        "need_research": decision.need_research,
        "mode": decision.mode,
        "queries": decision.queries[:4],
        "token_usage": [_token_usage(result["raw"], "Router")],
    }

def route_next(state: State) -> str:
    return "research" if state["need_research"] else "orchestrator"


def _tavily_search(query:str,max_result:int=5)->List[dict]:
    tool=TavilySearchResults(max_result=max_result)
    results=tool.invoke({'query':query})
    normalized: List[dict] = []
    for r in results or []:
        normalized.append(
            {
                "title": r.get("title") or "",
                "url": r.get("url") or "",
                "snippet": r.get("content") or r.get("snippet") or "",
                "published_at": r.get("published_date") or r.get("published_at"),
                "source": r.get("source"),
            }
        )
    return normalized

RESEARCH_SYSTEM = """You are a research synthesizer for technical writing.

Given raw web search results, produce a deduplicated list of EvidenceItem objects.

Rules:
- Only include items with a non-empty url.
- Prefer relevant + authoritative sources (company blogs, docs, reputable outlets).
- If a published date is explicitly present in the result payload, keep it as YYYY-MM-DD.
  If missing or unclear, set published_at=null. Do NOT guess.
- Keep snippets short.
- Deduplicate by URL.
"""

def research_node(state: State):
    # Search in parallel. Keep the number of queries bounded to control latency.
    queries = (state.get("queries", []) or [])[:4]
    if not queries:
        return {"evidence": []}

    def search_one(query):
        return _tavily_search(query, max_result=3)

    raw_results = []
    with ThreadPoolExecutor(max_workers=min(4, len(queries))) as pool:
        futures = [pool.submit(search_one, query) for query in queries]
        for future in as_completed(futures):
            try:
                raw_results.extend(future.result())
            except Exception:
                # One failed query should not cancel all other research.
                continue

    # Avoid a separate LLM call just to reformat search results.
    dedup = {}
    for item in raw_results:
        url = (item.get("url") or "").strip()
        if not url or url in dedup:
            continue

        dedup[url] = evidenceitem(
            title=(item.get("title") or "Untitled source")[:180],
            url=url,
            snippet=(item.get("snippet") or "")[:450],
            published_at=item.get("published_at"),
            source=item.get("source"),
        )

    # Keep the evidence pack compact for the planner and workers.
    evidence = list(dedup.values())[:10]
    return {"evidence": evidence}



ORCH_SYSTEM = """You are a senior technical writer and developer advocate.
Your job is to produce a highly actionable outline for a technical blog post.

Hard requirements:
- Create 5–6 focused sections (tasks) suitable for the topic and audience.
- Each task must include:
  1) goal (1 sentence)
  2) 3–6 bullets that are concrete, specific, and non-overlapping
  3) target word count (150–350)

Quality bar:
- Assume the reader is a developer; use correct terminology.
- Bullets must be actionable: build/compare/measure/verify/debug.
- Ensure the overall plan includes at least 2 of these somewhere:
  * minimal code sketch / MWE (set requires_code=True for that section)
  * edge cases / failure modes
  * performance/cost considerations
  * security/privacy considerations (if relevant)
  * debugging/observability tips

Grounding rules:
- Mode closed_book: keep it evergreen; do not depend on evidence.
- Mode hybrid:
  - Use evidence for up-to-date examples (models/tools/releases) in bullets.
  - Mark sections using fresh info as requires_research=True and requires_citations=True.
- Mode open_book:
  - Set blog_kind = "news_roundup".
  - Every section is about summarizing events + implications.
  - DO NOT include tutorial/how-to sections unless user explicitly asked for that.
  - If evidence is empty or insufficient, create a plan that transparently says "insufficient sources"
    and includes only what can be supported.

Output must strictly match the Plan schema.
"""

def orchestrator_node(state: State) -> dict:
    planner = model.with_structured_output(Plan, include_raw=True)

    evidence = state.get("evidence", [])
    mode = state.get("mode", "closed_book")

    result = planner.invoke([
        SystemMessage(content=ORCH_SYSTEM),
        HumanMessage(content=(
            f"Topic: {state['topic']}\\n"
            f"Mode: {mode}\\n"
            f"Evidence (only for fresh claims): "
            f"{[e.model_dump(exclude_none=True) for e in evidence[:8]]}"
        )),
    ])
    plan = result["parsed"]

    return {
        "plan": plan,
        "token_usage": [_token_usage(result["raw"], "Planner")],
    }



def fanout(state: State):
    plan = state["plan"]
    evidence = state.get("evidence", []) or []

    sends = []
    for task in plan.tasks:
        selected = _select_task_evidence(task, evidence, limit=4)
        sends.append(
            Send(
                "worker",
                {
                    "task": task.model_dump(),
                    "topic": state["topic"],
                    "mode": state.get("mode", "closed_book"),
                    "plan_context": {
                        "blog_title": plan.blog_title,
                        "audience": plan.audience,
                        "tone": plan.tone,
                        "blog_kind": plan.blog_kind,
                    },
                    "evidence": [
                        e.model_dump(exclude_none=True) for e in selected
                    ],
                },
            )
        )
    return sends



WORKER_SYSTEM = """You are a senior technical writer and developer advocate.
Write ONE section of a technical blog post in Markdown.

Hard constraints:
- Follow the provided Goal and cover ALL Bullets in order (do not skip or merge bullets).
- Stay close to Target words (±15%).
- Output ONLY the section content in Markdown (no blog title H1, no extra commentary).
- Start with a '## <Section Title>' heading.

Scope guard:
- If blog_kind == "news_roundup": do NOT turn this into a tutorial/how-to guide.
  Do NOT teach web scraping, RSS, automation, or "how to fetch news" unless bullets explicitly ask for it.
  Focus on summarizing events and implications.

Grounding policy:
- If mode == open_book:
  - Do NOT introduce any specific event/company/model/funding/policy claim unless it is supported by provided Evidence URLs.
  - For each event claim, attach a source as a Markdown link: ([Source](URL)).
  - Only use URLs provided in Evidence. If not supported, write: "Not found in provided sources."
- If requires_citations == true:
  - For outside-world claims, cite Evidence URLs the same way.
- Evergreen reasoning is OK without citations unless requires_citations is true.

Code:
- If requires_code == true, include at least one minimal, correct code snippet relevant to the bullets.

Style:
- Short paragraphs, bullets where helpful, code fences for code.
- Avoid fluff/marketing. Be precise and implementation-oriented.
"""



def worker_node(payload: dict) -> dict:
    task = Task(**payload["task"])
    plan = payload["plan_context"]
    evidence = [
        evidenceitem(**item) for item in payload.get("evidence", [])
    ]

    bullets_text = "\\n".join(f"- {bullet}" for bullet in task.bullets)
    evidence_text = "\\n".join(
        f"- {item.title} | {item.url}"
        + (f" | {item.snippet[:300]}" if item.snippet else "")
        for item in evidence[:4]
    )

    prompt = f"""Topic: {payload['topic']}
Title: {plan['blog_title']}
Audience: {plan['audience']}
Tone: {plan['tone']}
Blog kind: {plan['blog_kind']}
Mode: {payload.get('mode', 'closed_book')}

Section: {task.title}
Goal: {task.goal}
Target words: {task.target_words}
Tags: {', '.join(task.tags)}
Requires citations: {task.requires_citations}
Requires code: {task.requires_code}

Cover these points:
{bullets_text}

Evidence (cite only these URLs):
{evidence_text}
"""

    response = model.invoke([
        SystemMessage(content=WORKER_SYSTEM),
        HumanMessage(content=prompt),
    ])

    section_md = response.content.strip()
    usage = _token_usage(response, f"Worker-{task.id}")

    return {
        "sections": [(task.id, section_md)],
        "token_usage": [usage],
    }


####reducer subgraph

def merge_content(state: State) -> dict:

    plan = state["plan"]

    ordered_sections = [md for _, md in sorted(state["sections"], key=lambda x: x[0])]
    body = "\n\n".join(ordered_sections).strip()
    merged_md = f"# {plan.blog_title}\n\n{body}\n"
    return {"merged_md": merged_md}


DECIDE_IMAGES_SYSTEM = """You are an expert technical editor.
Decide if images/diagrams are needed for THIS blog.

Rules:
- Max 3 images total.
- Each image must materially improve understanding (diagram/flow/table-like visual).
- Insert placeholders exactly: [[IMAGE_1]], [[IMAGE_2]], [[IMAGE_3]].
- If no images needed: md_with_placeholders must equal input and images=[].
- Avoid decorative images; prefer technical diagrams with short labels.
Return strictly GlobalImagePlan.
"""

def decide_images(state: State) -> dict:

    planner = model.with_structured_output(GlobalImagePlan, include_raw=True)

    merged_md = state["merged_md"]
    plan = state["plan"]

    assert plan is not None

    result = planner.invoke([
        SystemMessage(content=DECIDE_IMAGES_SYSTEM),
        HumanMessage(content=(
            f"Blog kind: {plan.blog_kind}\\n"
            f"Topic: {state['topic']}\\n"
            f"Sections: {[task.title for task in plan.tasks]}\\n"
            f"Opening context: {merged_md[:1200]}"
        )),
    ])
    image_plan = result["parsed"]

    return {
        "image_specs": [img.model_dump() for img in image_plan.images],
        "token_usage": [_token_usage(result["raw"], "Image Planner")],
    }

def add_image_placeholders(state: State) -> dict:

    md = state["merged_md"]

    for image in state["image_specs"]:
        md += f"\n\n{image['placeholder']}\n"

    return {
        "md_with_placeholders": md
    }

def _gemini_generate_image_bytes(prompt: str) -> bytes:
    """
    Returns raw image bytes generated by Gemini.
    Requires: pip install google-genai
    Env var: GOOGLE_API_KEY
    """
    from google import genai
    from google.genai import types

    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("GOOGLE_API_KEY is not set.")

    client = genai.Client(api_key=api_key)

    resp = client.models.generate_content(
        model="gemini-2.5-flash-image",
        contents=prompt,
        config=types.GenerateContentConfig(
            response_modalities=["IMAGE"],
            safety_settings=[
                types.SafetySetting(
                    category="HARM_CATEGORY_DANGEROUS_CONTENT",
                    threshold="BLOCK_ONLY_HIGH",
                )
            ],
        ),
    )

    # Depending on SDK version, parts may hang off resp.candidates[0].content.parts
    parts = getattr(resp, "parts", None)
    if not parts and getattr(resp, "candidates", None):
        try:
            parts = resp.candidates[0].content.parts
        except Exception:
            parts = None

    if not parts:
        raise RuntimeError("No image content returned (safety/quota/SDK change).")

    for part in parts:
        inline = getattr(part, "inline_data", None)
        if inline and getattr(inline, "data", None):
            return inline.data

    raise RuntimeError("No inline image bytes found in response.")


def generate_and_place_images(state: State) -> dict:

    plan = state["plan"]
    assert plan is not None

    md = state.get("md_with_placeholders") or state["merged_md"]
    image_specs = state.get("image_specs", []) or []

    # If no images requested, just write merged markdown
    if not image_specs:
        filename = f"{plan.blog_title}.md"
        Path(filename).write_text(md, encoding="utf-8")
        return {"final": md}

    images_dir = Path("images")
    images_dir.mkdir(exist_ok=True)

    for spec in image_specs:
        placeholder = spec["placeholder"]
        filename = spec["filename"]
        out_path = images_dir / filename

        # generate only if needed
        if not out_path.exists():
            try:
                img_bytes = _gemini_generate_image_bytes(spec["prompt"])
                out_path.write_bytes(img_bytes)
            except Exception as e:
                # graceful fallback: keep doc usable
                prompt_block = (
                    f"> **[IMAGE GENERATION FAILED]** {spec.get('caption','')}\n>\n"
                    f"> **Alt:** {spec.get('alt','')}\n>\n"
                    f"> **Prompt:** {spec.get('prompt','')}\n>\n"
                    f"> **Error:** {e}\n"
                )
                md = md.replace(placeholder, prompt_block)
                continue

        img_md = f"![{spec['alt']}](images/{filename})\n*{spec['caption']}*"
        md = md.replace(placeholder, img_md)

    filename = f"{plan.blog_title}.md"
    Path(filename).write_text(md, encoding="utf-8")
    return {"final": md}

####subgraph
reducer_graph = StateGraph(State)

reducer_graph.add_node("merge_content", merge_content)
reducer_graph.add_node("decide_images", decide_images)
reducer_graph.add_node("add_image_placeholders", add_image_placeholders)
reducer_graph.add_node("generate_and_place_images", generate_and_place_images)

reducer_graph.add_edge(START, "merge_content")

reducer_graph.add_edge(
    "merge_content",
    "decide_images"
)

reducer_graph.add_edge(
    "decide_images",
    "add_image_placeholders"
)

reducer_graph.add_edge(
    "add_image_placeholders",
    "generate_and_place_images"
)

reducer_graph.add_edge(
    "generate_and_place_images",
    END
)

reducer_subgraph = reducer_graph.compile()

reducer_subgraph


##main graph
g = StateGraph(State)
g.add_node("router", router_node)
g.add_node("research", research_node)
g.add_node("orchestrator", orchestrator_node)
g.add_node("worker", worker_node)
g.add_node("reducer", reducer_subgraph)

g.add_edge(START, "router")
g.add_conditional_edges("router", route_next, {"research": "research", "orchestrator": "orchestrator"})
g.add_edge("research", "orchestrator")

g.add_conditional_edges("orchestrator", fanout, ["worker"])
g.add_edge("worker", "reducer")
g.add_edge("reducer", END)

app = g.compile()
app








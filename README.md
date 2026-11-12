# 🤖 LangGraph AI Blog Writer

An AI-powered blog generation application built using **LangGraph, Python, and Streamlit**. It automates the complete blog-writing process, from topic research and content planning to writing, image generation, and final Markdown output.

## 🚀 Features

- **AI-Powered Blog Generation:** Generate structured blogs from a user-provided topic.
- **Research Automation:** Gather relevant information and supporting evidence.
- **Intelligent Planning:** Create a blog outline and divide content into writing tasks.
- **Multi-Agent Workflow:** Use LangGraph to coordinate different AI agents.
- **Content Generation:** Generate detailed, well-structured blog articles.
- **Image Integration:** Generate and integrate relevant images into blogs.
- **Blog Archive:** Store generated blogs in an SQLite database.
- **Thread Management:** Associate generated blogs with thread IDs for organization.
- **Blog Search and Retrieval:** Search, access, and revisit previously generated blogs.
- **Markdown Export:** Download generated blogs in Markdown format.
- **ZIP Export:** Download blog content and associated resources together.
- **Streamlit Interface:** Interactive user interface for managing the blog-generation workflow.

## 🛠️ Tech Stack

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| LangGraph | Multi-agent workflow orchestration |
| LangChain | LLM integration |
| Groq API | AI-powered text generation |
| Google Gemini API | AI and image-related capabilities |
| Streamlit | Frontend interface |
| SQLite | Persistent blog storage |
| Markdown | Blog content formatting |

## 📂 Project Structure

```text
LangGraph-AI-Blog-Writer/
│
├── frontend.py          # Streamlit application
├── workflow.py          # LangGraph workflow and agents
├── blog_store.py        # SQLite blog archive management
├── requirements.txt     # Project dependencies
├── .env.example         # Environment variable template
├── .gitignore           # Git ignored files
└── README.md            # Project documentation
```

## ⚙️ Installation and Setup

### 1. Clone the repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd LangGraph-AI-Blog-Writer
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

### 3. Activate the environment

**Windows:**

```bash
venv\Scripts\activate
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

### 5. Configure API keys

Create a `.env` file in the project root directory.

```env
GROQ_API_KEY=your_groq_api_key
GOOGLE_API_KEY=your_google_api_key
```

Replace the placeholder values with your actual API keys.

**Important:** Never upload your `.env` file or expose API keys on GitHub.

### 6. Run the application

```bash
streamlit run frontend.py
```

The application will open in your browser.

## 💡 How It Works

1. Enter a blog topic in the Streamlit application.
2. Start the blog-generation process.
3. The LangGraph workflow processes the topic.
4. Research agents collect relevant information.
5. The planner creates a structured blog outline.
6. Writing agents generate individual blog sections.
7. The workflow combines the sections into a final blog.
8. Relevant images are integrated into the content.
9. The generated blog is saved in the SQLite archive.
10. Access previously generated blogs whenever needed.

## 💾 Blog Storage

The project uses SQLite to maintain a persistent archive of generated blogs.

Each saved blog can contain:

- Unique blog ID
- Thread ID
- Blog title
- Topic
- Markdown content
- Image specifications
- Creation timestamp

This allows users to retrieve previously generated content without generating it again.

## 🔐 Environment Variables

| Variable | Description |
|---|---|
| `GROQ_API_KEY` | API key for Groq |
| `GOOGLE_API_KEY` | API key for Google AI services |

## 🔮 Future Improvements

- User authentication and personalized blog libraries.
- Support for additional LLM providers.
- Blog editing and regeneration.
- PDF and DOCX export.
- SEO optimization and keyword analysis.
- Deployment with cloud-based database storage.

## 👨‍💻 Author

**Omkar Chavan**

AI Engineer | Python Developer | Generative AI Enthusiast

---

⭐ If you find this project useful, consider giving it a star on GitHub!

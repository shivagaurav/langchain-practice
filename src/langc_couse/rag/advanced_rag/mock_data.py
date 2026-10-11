from langchain_core.documents import Document

practice_docs = [
    # --- PROGRAMMING & TECH ---
    # Good for practicing basic similarity search and categorical filtering
    Document(
        page_content="python is a highlevel programming language known for it's simplicity",
        metadata={
            "topic": "programming",
            "language": "python",
            "difficulty": "beginner",
            "year_updated": 2024,
        },
    ),
    Document(
        page_content="JavaScript is a versatile scripting language primarily used for dynamic web content and frontend development.",
        metadata={
            "topic": "programming",
            "language": "javascript",
            "difficulty": "beginner",
            "year_updated": 2023,
        },
    ),
    Document(
        page_content="LangChain is a framework for developing applications powered by language models, focusing on chaining components, memory, and agents.",
        metadata={
            "topic": "artificial intelligence",
            "framework": "langchain",
            "difficulty": "intermediate",
            "year_updated": 2024,
        },
    ),
    Document(
        page_content="Rust enforces memory safety without a garbage collector via its strict ownership and borrowing model.",
        metadata={
            "topic": "programming",
            "language": "rust",
            "difficulty": "advanced",
            "year_updated": 2023,
        },
    ),
    Document(
        page_content="Vector databases like Chroma and Pinecone store embeddings to enable fast similarity search for RAG pipelines.",
        metadata={
            "topic": "data engineering",
            "tool": "vector_database",
            "difficulty": "intermediate",
            "year_updated": 2024,
        },
    ),
    # --- CORPORATE / HR POLICIES ---
    # Good for practicing department-level access control and exact match filters ($eq)
    Document(
        page_content="Employees are entitled to 20 days of paid time off per calendar year. Unused PTO does not roll over to the next year.",
        metadata={
            "category": "hr_policy",
            "department": "all",
            "doc_type": "handbook",
            "confidentiality": "internal",
        },
    ),
    Document(
        page_content="Engineering team members can expense up to $500 annually for cloud computing courses and certifications.",
        metadata={
            "category": "benefits",
            "department": "engineering",
            "doc_type": "guideline",
            "confidentiality": "internal",
        },
    ),
    Document(
        page_content="The Q3 marketing budget has been increased by 15% to support the upcoming European product launch.",
        metadata={
            "category": "internal_memo",
            "department": "marketing",
            "doc_type": "report",
            "year": 2025,
        },
    ),
    # --- FINANCIAL & BUSINESS ---
    # Good for practicing numeric comparison filters like greater-than ($gt) or less-than ($lt)
    Document(
        page_content="Company revenue grew by 24% year-over-year in 2024, largely driven by enterprise software subscriptions.",
        metadata={
            "topic": "finance",
            "report_type": "annual_earnings",
            "year": 2024,
            "confidentiality": "public",
        },
    ),
    Document(
        page_content="Projected R&D spending for Q1 2025 is estimated at $4.2 million, prioritizing generative AI integration.",
        metadata={
            "topic": "finance",
            "report_type": "projections",
            "year": 2025,
            "confidentiality": "internal",
        },
    ),
    Document(
        page_content="Q1 2023 earnings were heavily impacted by supply chain disruptions, resulting in a 5% revenue drop.",
        metadata={
            "topic": "finance",
            "report_type": "quarterly_earnings",
            "year": 2023,
            "confidentiality": "public",
        },
    ),
    # --- CUSTOMER SUPPORT TICKETS ---
    # Good for hybrid search (combining keyword search for ticket status with semantic search for the issue)
    Document(
        page_content="User reports the mobile app crashes immediately upon opening on iOS 17. Issue reproduced by QA team.",
        metadata={
            "source": "zendesk",
            "ticket_id": "TCK-8921",
            "status": "open",
            "priority": "high",
            "tags": ["crash", "ios", "mobile"],
        },
    ),
    Document(
        page_content="Customer requesting a refund for their annual subscription. Stated they no longer need the service.",
        metadata={
            "source": "zendesk",
            "ticket_id": "TCK-8922",
            "status": "closed",
            "priority": "low",
            "tags": ["billing", "refund"],
        },
    ),
    Document(
        page_content="API rate limits are triggering too early for Enterprise tier users. Temporary workaround provided.",
        metadata={
            "source": "jira",
            "ticket_id": "ENG-402",
            "status": "in_progress",
            "priority": "critical",
            "tags": ["api", "enterprise", "bug"],
        },
    ),
]

from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    CharacterTextSplitter,
    TokenTextSplitter,
    MarkdownHeaderTextSplitter,
    Language
)

from langchain_core.documents import Document

from dotenv import load_dotenv

load_dotenv()


# Sample documents for testing
SAMPLE_TEXT = """# Introduction to Machine Learning

Machine learning is a subset of artificial intelligence that enables systems to learn and improve from experience without being explicitly programmed.

## Types of Machine Learning

### Supervised Learning
Supervised learning uses labeled data to train models. The algorithm learns to map inputs to outputs based on example input-output pairs.

Common algorithms include:
- Linear Regression
- Decision Trees
- Neural Networks

### Unsupervised Learning
Unsupervised learning finds hidden patterns in unlabeled data. The algorithm discovers structure without predefined labels.

Common algorithms include:
- K-Means Clustering
- Principal Component Analysis
- Autoencoders

## Applications

Machine learning is used in many fields:
1. Image recognition
2. Natural language processing
3. Recommendation systems
4. Fraud detection
5. Autonomous vehicles
""".strip()

SAMPLE_CODE = '''
def quicksort(arr):
    """
    Quicksort implementation in Python.
    Time complexity: O(n log n) average, O(n²) worst case.
    """
    if len(arr) <= 1:
        return arr

    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]

    return quicksort(left) + middle + quicksort(right)


def binary_search(arr, target):
    """
    Binary search implementation.
    Requires sorted array.
    Time complexity: O(log n)
    """
    left, right = 0, len(arr) - 1

    while left <= right:
        mid = (left + right) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1

    return -1
'''

def recursive_splitter():
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", " ", ""],
    )

    chunks = splitter.split_text(SAMPLE_TEXT)

    print(f"the number of chunks {len(chunks)}")

    for i, c in enumerate(chunks):
        print(f".......................CHUNK {i+1} START....................")
        print("************************************************************")
        print(c)
        print("************************************************************")
        print(f".......................CHUNK {i+1} END......................")

# recursive_splitter()

def chunk_importance():
    TEXT = "API key expires in 24 hours, please refresh on time using the API end point in order avoid getting logged out"

    no_overlap = RecursiveCharacterTextSplitter(chunk_size=50, chunk_overlap=0)
    with_overlap = RecursiveCharacterTextSplitter(chunk_size=50, chunk_overlap=20)

    no_overlap_chunks = no_overlap.split_text(TEXT)
    with_overlap_chunks = with_overlap.split_text(TEXT)

    print("no overlap....... len ", len(no_overlap_chunks))
    print('first chunk........  ', no_overlap_chunks[0])
    print("***********************************************************")
    print('second chunk........  ', no_overlap_chunks[1])
    print("***********************************************************")
    print('third chunk........  ', no_overlap_chunks[2])
    print("*****************************************END************************************")
    print("*****************************************END************************************")


    print("with overlap....... len ", len(with_overlap_chunks))
    print('first chunk........  ', with_overlap_chunks[0])
    print("***********************************************************")
    print('second chunk........  ', with_overlap_chunks[1])
    print("***********************************************************")
    print('third chunk........  ', with_overlap_chunks[2])

# chunk_importance()

def markdown_splitter():
    headers_to_consider = [
        ("#", "h1"),
        ("##", "h2"),
        ("###", "h3")
    ]

    splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_consider)

    chunks = splitter.split_text(SAMPLE_TEXT)

    print("number of chunks loaded ", len(chunks))

    for c in chunks:
        print("*************************************CHUNK START***************************")
        print(c.page_content)
        print("*************************************CHUNK END*****************************")

# markdown_splitter()

def code_splitter():
    splitter = RecursiveCharacterTextSplitter.from_language(chunk_size = 500, chunk_overlap=50, language=Language.PYTHON)

    chunks = splitter.split_text(SAMPLE_CODE)

    print("no. of code chunks generated: ", len(chunks))

    for c in chunks:
        print("CHUNK START**************************************************")
        print(c)
        print("CHUNK END****************************************************")

# code_splitter()

def document_splitter():
    from langchain_community.document_loaders import PyPDFLoader

    loader = PyPDFLoader("./docs/langchain_demo.pdf")
    docs = loader.load()

    print(f"loaded {len(docs)} doc(s)")

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

    chunks = splitter.split_documents(docs)

    print(f"split into {len(chunks)} chunk(s)")

    print(f"meta data for one chunk {chunks[0].metadata}")

    print(f"meta data for one chunk {chunks[0].page_content}")

document_splitter()


# Crawler Documentation

This repository provides a web crawling solution designed for scraping structured JSON data from websites using a multi-agent architecture. The `Crawler` class extracts, refines, processes, and aggregates data into a unified output, ideal for integration into Retrieval-Augmented Generation (RAG) pipelines.

---

## Features

1. **Multi-Agent Architecture**  
   - **Agent 1:** Explores and discovers routes (URLs) on the target website.  
   - **Agent 2:** Refines the list of routes to focus on those most relevant to the task.  
   - **Agent 3:** Extracts JSON data from each refined URL using content relevance filtering and scraping.  
   - **Agent 4:** Aggregates all scraped data into a coherent JSON result.

2. **Context-Aware Processing**  
   - Leverages TF-IDF and cosine similarity to extract relevant documents for scraping.  
   - Utilizes OpenAI’s LLM for refining routes and interpreting HTML content.  

3. **Error Handling**  
   - Validates and repairs JSON using the `partialjson` library.  
   - Handles nested and incomplete JSON structures robustly.

4. **Extensible Output Formats**  
   - Currently supports JSON output with straightforward extensibility for other formats.

---

## How the Crawler Works

### Initialization
The `Crawler` is initialized with parameters including base URL, scraping instructions, and output format. The `depth` parameter determines how many levels of links the crawler explores.

```python
crawler = Crawler(
    instructions="Find FAQS and answers to FAQS",
    base_url="https://shop.westcoastcustoms.com",
    api_key="your_openai_api_key",
    depth=1,
    output="json"
)
```

### Execution
The `execute()` method orchestrates the multi-agent workflow:
1. Fetches the main webpage.
2. Discovers all routes on the site, recursively exploring links based on depth.
3. Refines the list of URLs based on task-specific instructions.
4. Scrapes data from refined routes and filters relevant information.
5. Aggregates the results into a unified JSON object.

```python
output = crawler.execute().content
print(output)
```

---

## Integration into a RAG Pipeline

This crawler can be seamlessly integrated into a Retrieval-Augmented Generation (RAG) pipeline to enhance data retrieval and knowledge generation:

1. **Document Retrieval**  
   Use the crawler to extract JSON documents directly from relevant sections of a website.

2. **Embedding Generation**  
   Employ embeddings (e.g., OpenAI Embeddings) to index and retrieve documents based on semantic similarity.

3. **Knowledge Augmentation**  
   Provide the extracted data to an LLM as context for answering questions or generating insights.

### Example: RAG Workflow

```python
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS

# Initialize the Crawler
crawler = Crawler("Find product details and pricing", "https://example.com", api_key, depth=2, output="json")
data = json.loads(crawler.execute().content)

# Process Data into Documents
documents = [{"content": item, "metadata": {}} for item in data]

# Generate Embeddings
embeddings = OpenAIEmbeddings()
vector_store = FAISS.from_documents(documents, embeddings)

# Query the Vector Store
query = "What is the price of product X?"
results = vector_store.similarity_search(query, k=5)
```

---

## Example Usage

### Basic Script

```python
from crawler import Crawler
from dotenv import load_dotenv
import os
import json

# Load API Key
load_dotenv()
api_key = os.getenv("OPEN_AI_API_KEY")

# Initialize and Execute the Crawler
client = Crawler(
    "Find FAQS and answers to FAQS",
    "https://shop.westcoastcustoms.com",
    api_key,
    depth=1,
    output="json"
)
output = client.execute().content

# Save Output
json.dump(json.loads(output), open("output.json", "w"))
```

### Output Example

```json
{
    "FAQs": [
        {
            "question": "Do you accept walk-in's?",
            "answer": "Unfortunately not, we work on a strict schedule on things that are planned months in advance, we advise you to fill out an inquiry form if you're looking to get something done with us and we will call you back if we think its something we can take on!"
        },
        {
            "question": "Do you do sponsorships?",
            "answer": "Unfortunately we don't."
        },
        {
            "question": "Do you accept oversea's jobs?",
            "answer": "Yes! We will add the cost of shipping into the final price."
        },
        {
            "question": "How do i get on the show?",
            "answer": "All candidates are chosen by a third party, we don't have a say on who gets chosen."
        },
        {
            "question": "Do you work on non-celebrity cars?",
            "answer": "Yes! we accept jobs from everyone!"
        },
        {
            "question": "Can i get a quote on my vehicle customization?",
            "answer": "Yes! Just fill out this form!"
        },
        {
            "question": "Can i hold an event at WCC?",
            "answer": "Yes! We accept events just fill out this form and we will get back to you!"
        },
        {
            "question": "Where can I watch the show's?",
            "answer": "Pimp my Ride finished in 2004 but can still be found on Amazon Prime Video! Inside West Coast Customs can be watched on Netflix! Our new show Pimped Out can be watched on our YouTube Channel!"
        },
        {
            "question": "Can I buy merch in-store?",
            "answer": "Yes! We have our walk-in Merch Room where you can buy in-store 2101 W. Empire Ave Burbank CA, 91504. OR place a pickup order from our website!"
        },
        {
            "question": "Do you buy cars?",
            "answer": "No, unfortunately we don't."
        },
        {
            "question": "Ready to start?",
            "answer": "Send Inquiry"
        },
        {
            "question": "Business wanting to collab?",
            "answer": "Email Us"
        },
        {
            "question": "Have a question?",
            "answer": "FAQ"
        }
    ]
}
```

---

## Dependencies


Install all dependencies using pip:

```bash
pip install -r requirements.txt
```

---
## Running the WebUI

To launch the WebUI run
```bash
python3 ui_for_app.py
```

---
---

## Future Enhancements

- Support for additional output formats (e.g., CSV).  
- Parallelized route exploration for improved efficiency.  
- Enhanced error handling for non-standard HTML structures.  

---

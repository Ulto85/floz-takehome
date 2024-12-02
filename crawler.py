import requests
from bs4 import BeautifulSoup
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI
import csv
import ast
import json
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import AsyncHtmlLoader
from langchain_community.document_transformers import Html2TextTransformer
from langchain.embeddings import OpenAIEmbeddings
from langchain.document_loaders import TextLoader, CSVLoader, JSONLoader
from partialjson.json_parser import JSONParser
parser = JSONParser()
from dotenv import load_dotenv
load_dotenv()
class CrawlerResult:
    def __init__(self, content):
        """
        Initialize CrawlerResult with JSON validation.
        
        Args:
            content: Raw scraped content
        """
        # Attempt to parse and validate JSON
        self.content = self._validate_json(content)

    def _validate_json(self, content):
        """
        Validate and repair JSON content.

        Args:
            content: Input content to validate.

        Returns:
            Valid JSON string or an empty JSON array (if repair fails).
        """
 
        if isinstance(content, (dict, list)):
            return json.dumps(content, indent=2)

        content = str(content).strip()

        content = re.sub(r'^\s*\w+\s*{', '{', content)


        try:

            json.loads(content)
            return content
        except json.JSONDecodeError:
            pass

 
        try:
            parsed=parser.parse(content)
            
           
            return json.dumps(parsed)
        except json.JSONDecodeError:
        
            return json.dumps([], indent=2)

    
    def __str__(self):
        """
        String representation of the content.
        
        Returns:
            Validated JSON string
        """
        return self.content

    def __repr__(self):
        """
        Representation of the CrawlerResult.
        
        Returns:
            Representation of the content
        """
        return self.content
class Crawler:
    def __init__(self, instructions, base_url, api_key, depth=1,output="json"):
        """
        Initializes the Crawler instance.

        Args:
            instructions (str): The instructions for scraping.
            base_url (str): The base URL to start scraping from.
            output (str): The desired output format (support only "json")
        """
        self.instructions = instructions
        self.base_url = base_url.rstrip("/") 
        self.output = output.lower()
        self.api_key = api_key
        self.llm = OpenAI(model_name="gpt-3.5-turbo-16k",temperature=0,api_key = self.api_key,max_tokens=10000)
        self.routes = []  
        self.data = []
        self.depth=depth

    def fetch_website(self, url):
        """Fetch the HTML content of the website."""
        try:
            response = requests.get(url)
            response.raise_for_status()
            return response.text
        except requests.RequestException as e:
            return f"Error fetching {url}: {str(e)}"

    def explore_routes(self, html,depth):
        """Extract all unique routes (e.g., subpage URLs) from the main page."""
        soup = BeautifulSoup(html, "html.parser")
        base_domain = self.base_url
        routes = set()
        for link in soup.find_all("a", href=True):
            href = link["href"]
            if href.startswith("/"):  
                routes.add(base_domain + href)
            elif href.startswith(base_domain):  
                routes.add(href)
        answer = list(routes)
        if depth-1:
            for r in routes:
                if r != base_domain:
                    answer.extend(self.explore_routes(self.fetch_website(r),depth-1))
        return list(set(answer))

    def refine_routes(self, routes):
        """Agent 2: Refine the list of routes based on the instructions."""
        refine_prompt = PromptTemplate(
            input_variables=["instructions", "routes"],
            template=(
                "You are given the following list of routes: {routes}. Based on the instructions: {instructions}, "
                "select the 3 routes that are most relevant for data scraping."
            )
        )
        chain = LLMChain(llm=self.llm, prompt=refine_prompt)
        llm_response = chain.run({"instructions": self.instructions, "routes": routes})
        url_pattern = re.escape(self.base_url) + r"[^\s'\"]*"  
        refined_routes = re.findall(url_pattern, llm_response)
        return refined_routes

    def scrape_page(self, html, refined_instructions):
        """
        Enhanced scraping method with document relevance filtering
        
        Args:
            html (str): HTML content to process
            refined_instructions (str): Specific scraping instructions
        
        Returns:
            Scraped and processed data
        """
       
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=4000,
            chunk_overlap=200
        )
        chunks = text_splitter.split_text(html)
        
    
        relevant_chunks = self.extract_relevant_documents(
            chunks, 
            refined_instructions, 
            top_k=3
        )
        
 
        if not relevant_chunks:
            return []

        relevant_html = "\n".join(relevant_chunks)
       
    
        scrape_prompt = PromptTemplate(
            input_variables=["refined_instructions", "html"],
            template=(
                "You are provided with the following instructions: {refined_instructions}, "
                "and the content from the website: {html}. "
                "EXTRACT DATA precisely as per the instructions and provide it in " + self.output + " format. "
                "Be concise and focus only on the most relevant information."
            )
        )
        chain = LLMChain(llm=self.llm, prompt=scrape_prompt)
        return chain.run({
            "refined_instructions": refined_instructions, 
            "html": relevant_html
        })

    def save_output(self):
        """Save the scraped data in the desired format."""
        if self.output == "csv":
            with open("output.csv", "w", newline="") as file:
                writer = csv.writer(file)
                if self.data and isinstance(self.data[0], dict):
                    writer.writerow(self.data[0].keys())  # Header row
                    for row in self.data:
                        writer.writerow(row.values())
        elif self.output == "json":
            with open("output.json", "w") as file:
                json.dump(self.data, file, indent=4)
    def aggregate_results(self):
        """Agent to aggregate all scraped data into a single result."""
        aggregation_prompt = PromptTemplate(
            input_variables=["data"],
            template=(
                "You are given the following data collected from multiple routes: {data}. "
                "Aggregate this data into a single, cohesive output that is well-organized and easy to understand. Only include information that was relevant to the initial task: {task}" 
                "Return the aggregated result as a " + self.output
            )
        )
        chain = LLMChain(llm=self.llm, prompt=aggregation_prompt)
        aggregated_result = chain.run({"data": self.data,"task":self.instructions})
        return aggregated_result

    def execute(self):
        """Main function to execute the multi-agent scraping process."""
        print(f"Step 1: Fetching the main page: {self.base_url}")
        main_page_html = self.fetch_website(self.base_url)
      
        print("Step 2: Exploring routes...")
        routes = self.explore_routes(main_page_html,self.depth)

       
        if self.base_url not in routes:
            print(f"Adding base URL to the list of routes: {self.base_url}")
            routes.append(self.base_url)

        print(f"Discovered Routes:\n{routes}")
        print(routes)
   
        print("Step 3: Refining routes based on instructions...")
        refined_routes = list(set(self.refine_routes(routes)))
        print(f"Refined Routes:\n{refined_routes}")
  

        print("Step 4: Scraping data from refined routes...")
        loader = AsyncHtmlLoader(refined_routes)
        docs = loader.load()
        html2text = Html2TextTransformer()
        docs = html2text.transform_documents(docs)
        
        for doc in docs:
            
            scraped_data = self.scrape_page( doc.page_content, self.instructions)
            print(f"Scraped Data:\n{scraped_data}")
            
            if isinstance(scraped_data, list):
                self.data.extend(scraped_data)
            else:
                self.data.append(scraped_data)

        print("Step 5: Aggregating results...")
        aggregated_result = self.aggregate_results()
        print(f"Aggregated Result:\n{aggregated_result}")

        
        return CrawlerResult(aggregated_result)
    def extract_relevant_documents(self, documents, instructions, top_k=3):
        """
        Extract most relevant documents using TF-IDF and cosine similarity
        
        Args:
            documents (list): List of document contents
            instructions (str): Scraping instructions
            top_k (int): Number of top relevant documents to return
        
        Returns:
            list: Most relevant document contents
        """
        
        if not documents:
            return []
        
        corpus = documents + [instructions]
        

        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(corpus)
        
        similarity_scores = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1])[0]
        
        top_indices = similarity_scores.argsort()[-top_k:][::-1]
        
    
        self.relevant_docs = [documents[idx] for idx in top_indices]
        return self.relevant_docs


   

    
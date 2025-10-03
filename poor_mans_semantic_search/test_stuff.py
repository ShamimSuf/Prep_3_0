import json
import re
from typing import Dict, List
import spacy
import nltk
from nltk.corpus import wordnet

class DatasetProcessing:
    """Class for processing and tokenizing datasets"""
    
    def __init__(self):
        # Load spaCy model for stop word removal
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            print("spaCy model 'en_core_web_sm' not found. Install with: python -m spacy download en_core_web_sm")
            self.nlp = None
    
    def tokenize_text(self, text: str) -> List[str]:
        """
        Tokenize text by:
        1. Converting to lowercase
        2. Removing punctuation
        3. Splitting into words
        4. Filtering out empty strings
        """
        # Convert to lowercase and remove punctuation
        text = re.sub(r'[^\w\s]', '', text.lower())
        # Split into tokens and filter empty strings
        tokens = [token for token in text.split() if token.strip()]
        return tokens
    
    def load_and_tokenize_dataset(self, json_file_path: str) -> Dict[str, List[str]]:

        try:
            with open(json_file_path, 'r', encoding='utf-8') as file:
                data = json.load(file)
            
            tokenized_docs = {}
            
            # Process each document in the dataset
            for doc in data['documents']:
                doc_id = doc['id']
                text = doc['text']
                
                # Tokenize only the text field
                tokens = self.tokenize_text(text)
                tokenized_docs[doc_id] = tokens
            
            return tokenized_docs
            
        except FileNotFoundError:
            print(f"Error: File {json_file_path} not found")
            return {}
        except json.JSONDecodeError:
            print(f"Error: Invalid JSON in file {json_file_path}")
            return {}
        except KeyError as e:
            print(f"Error: Missing key {e} in JSON structure")
            return {}
    
    def remove_stop_words(self, tokenized_docs: Dict[str, List[str]]) -> Dict[str, List[str]]:

        if self.nlp is None:
            print("Warning: spaCy model not loaded. Returning original tokens.")
            return tokenized_docs
        
        filtered_docs = {}
        
        for doc_id, tokens in tokenized_docs.items():
            # Filter out stop words using spaCy's stop word list
            filtered_tokens = []
            for token in tokens:
                if token not in self.nlp.Defaults.stop_words:
                    filtered_tokens.append(token)
            filtered_docs[doc_id] = filtered_tokens
        
        return filtered_docs
    
    def lemmatize_tokens(self, tokenized_docs: Dict[str, List[str]]) -> Dict[str, List[str]]:

        if self.nlp is None:
            print("Warning: spaCy model not loaded. Returning original tokens.")
            return tokenized_docs
        
        lemmatized_docs = {}
        
        for doc_id, tokens in tokenized_docs.items():
            # Lemmatize each token individually
            lemmatized_tokens = []
            for token in tokens:
                # Process individual token to get its lemma
                doc = self.nlp(token)
                lemmatized_tokens.append(doc[0].lemma_)
            
            lemmatized_docs[doc_id] = lemmatized_tokens
        
        return lemmatized_docs
    
    def build_inverted_index(self, lemmatized_docs: Dict[str, List[str]]) -> Dict[str, Dict[str, int]]:
        """
        Build an inverted index from lemmatized documents.
        
        Args:
            lemmatized_docs: Dict where key is doc_id and value is array of lemmatized tokens
            
        Returns:
            Inverted index dict where:
            - Key is the term (lemmatized word)
            - Value is dict of {doc_id: count} for documents containing that term
        """
        inverted_index = {}
        
        for doc_id, tokens in lemmatized_docs.items():
            # Count frequency of each token in this document
            token_counts = {}
            for token in tokens:
                if token in token_counts:
                    token_counts[token] += 1
                else:
                    token_counts[token] = 1
            
            # Add to inverted index
            for token, count in token_counts.items():
                if token not in inverted_index:
                    inverted_index[token] = {}
                inverted_index[token][doc_id] = count
        
        return inverted_index
    
    def get_synonyms_from_wordnet(self, inverted_index: Dict[str, Dict[str, int]]) -> Dict[str, List[str]]:
        """
        Get synonyms for each term in the inverted index using WordNet.
        
        Args:
            inverted_index: Dict where key is term and value is dict of {doc_id: count}
            
        Returns:
            Dict where key is the lemma term and value is array of synonyms (max 10)
        """
        try:
            # Download WordNet data if not already present
            nltk.download('wordnet', quiet=True)
            nltk.download('omw-1.4', quiet=True)
        except Exception:
            print("Warning: Could not download WordNet data. Install NLTK and try again.")
            return {}
        
        synonyms_dict = {}
        
        for term in inverted_index.keys():
            synonyms = set()
            
            # Get all synsets (synonym sets) for this word
            synsets = wordnet.synsets(term)
            
            for synset in synsets:
                # Get lemmas (word forms) from each synset
                for lemma in synset.lemmas():
                    synonym = lemma.name().lower()
                    # Only include single words (no underscores or spaces)
                    if '_' not in synonym and ' ' not in synonym and synonym != term and len(synonym) > 1:
                        synonyms.add(synonym)
            
            # Convert to list and limit to 10 synonyms
            synonym_list = list(synonyms)[:10]
            synonyms_dict[term] = synonym_list
        
        return synonyms_dict


class SynonymBasedTFIDF:
    """Class for calculating synonym-based TF-IDF scores"""
    
    def __init__(self, lemmatized_docs: Dict[str, List[str]], 
                 inverted_index: Dict[str, Dict[str, int]], 
                 synonyms_dict: Dict[str, List[str]]):
        self.lemmatized_docs = lemmatized_docs
        self.inverted_index = inverted_index
        self.synonyms_dict = synonyms_dict
        self.N = len(lemmatized_docs)  # Total number of documents
    
    def get_synonym_set(self, term: str) -> set:
        """
        Get synonym set S = {term} ∪ synonyms for a given term
        
        Args:
            term: the query term
            
        Returns:
            Set containing the term and its synonyms
        """
        synonym_set = {term}  # Start with the term itself
        
        # Add synonyms if they exist
        if term in self.synonyms_dict:
            synonym_set.update(self.synonyms_dict[term])
        
        return synonym_set
    
    def calculate_tf_syn(self, synonym_set: set, doc_id: str) -> float:
        """
        Calculate normalized TF_syn(S, doc) = (Σ count(s, doc) for all s ∈ S) / (total tokens in doc)
        
        Args:
            synonym_set: Set of synonyms including the original term
            doc_id: Document ID
            
        Returns:
            Normalized synonym-based term frequency
        """
        synonym_count = 0
        
        # Sum counts of all synonyms in the document
        for synonym in synonym_set:
            if synonym in self.inverted_index and doc_id in self.inverted_index[synonym]:
                synonym_count += self.inverted_index[synonym][doc_id]
        
        # Get total number of tokens in the document for normalization
        total_tokens = len(self.lemmatized_docs[doc_id]) if doc_id in self.lemmatized_docs else 1
        
        # Normalize by total tokens in document
        tf_syn_normalized = synonym_count / total_tokens
        
        return tf_syn_normalized
    
    def calculate_df_syn(self, synonym_set: set) -> int:
        """
        Calculate df_syn(S, corpus) = |{d ∈ corpus : ∃s ∈ S, s ∈ d}|
        
        Args:
            synonym_set: Set of synonyms including the original term
            
        Returns:
            Number of documents containing at least one synonym from the set
        """
        docs_with_synonyms = set()
        
        for synonym in synonym_set:
            if synonym in self.inverted_index:
                # Add all document IDs that contain this synonym
                docs_with_synonyms.update(self.inverted_index[synonym].keys())
        
        return len(docs_with_synonyms)
    
    def calculate_idf_syn(self, synonym_set: set) -> float:
        """
        Calculate IDF_syn(S, corpus) = log((N + 1) / (df_syn(S, corpus) + 1))
        
        Args:
            synonym_set: Set of synonyms including the original term
            
        Returns:
            Synonym-based inverse document frequency
        """
        import math
        
        df_syn = self.calculate_df_syn(synonym_set)
        idf_syn = math.log((self.N + 1) / (df_syn + 1))
        
        return idf_syn
    
    def calculate_tfidf_syn(self, term: str, doc_id: str) -> float:
        """
        Calculate TF-IDF_syn(S, doc, corpus) = TF_syn(S, doc) × IDF_syn(S, corpus)
        
        Args:
            term: the query term
            doc_id: Document ID
            
        Returns:
            Synonym-based TF-IDF score
        """
        synonym_set = self.get_synonym_set(term)
        tf_syn = self.calculate_tf_syn(synonym_set, doc_id)
        idf_syn = self.calculate_idf_syn(synonym_set)
        
        return tf_syn * idf_syn
    
    def get_term_scores_for_document(self, doc_id: str, terms: List[str]) -> Dict[str, float]:
        """
        Calculate synonym-based TF-IDF scores for multiple terms in a document
        
        Args:
            doc_id: Document ID
            terms: List of terms to calculate scores for
            
        Returns:
            Dictionary mapping terms to their synonym-based TF-IDF scores
        """
        scores = {}
        
        for term in terms:
            scores[term] = self.calculate_tfidf_syn(term, doc_id)
        
        return scores
    
    def show_term_analysis(self, term: str, doc_id: str):
        """
        Show detailed analysis of synonym-based TF-IDF calculation for a term
        
        Args:
            term: the query term
            doc_id: Document ID
        """
        synonym_set = self.get_synonym_set(term)
        tf_syn = self.calculate_tf_syn(synonym_set, doc_id)
        df_syn = self.calculate_df_syn(synonym_set)
        idf_syn = self.calculate_idf_syn(synonym_set)
        tfidf_syn = tf_syn * idf_syn
        
        print(f"\n--- Synonym-based TF-IDF Analysis for '{term}' in '{doc_id}' ---")
        print(f"Synonym set S: {synonym_set}")
        print(f"TF_syn(S, {doc_id}): {tf_syn}")
        print(f"df_syn(S, corpus): {df_syn} documents")
        print(f"IDF_syn(S, corpus): {idf_syn:.4f}")
        print(f"TF-IDF_syn(S, {doc_id}, corpus): {tfidf_syn:.4f}")


class PoorMansSemanticSearch:
    """Complete Poor Man's Semantic Search implementation"""
    
    def __init__(self, dataset_path: str = None):
        self.processor = DatasetProcessing()
        self.tfidf_calculator = None
        self.original_docs = {}
        
        if dataset_path:
            self.load_dataset(dataset_path)
    
    def load_dataset(self, dataset_path: str):
        """Load and process the dataset for searching"""
        print("Loading and processing dataset...")
        
        # Load original documents for display
        try:
            with open(dataset_path, 'r', encoding='utf-8') as file:
                data = json.load(file)
                self.original_docs = {doc['id']: doc for doc in data['documents']}
        except Exception as e:
            print(f"Error loading original documents: {e}")
            return False
        
        # Process through the full pipeline
        tokenized_docs = self.processor.load_and_tokenize_dataset(dataset_path)
        if not tokenized_docs:
            return False
            
        filtered_docs = self.processor.remove_stop_words(tokenized_docs)
        lemmatized_docs = self.processor.lemmatize_tokens(filtered_docs)
        inverted_index = self.processor.build_inverted_index(lemmatized_docs)
        synonyms_dict = self.processor.get_synonyms_from_wordnet(inverted_index)
        
        # Initialize TF-IDF calculator
        self.tfidf_calculator = SynonymBasedTFIDF(lemmatized_docs, inverted_index, synonyms_dict)
        
        return True
    
    def process_query(self, query_text: str) -> list:
        """
        Process query text into lemmatized terms
        
        Args:
            query_text: Raw query string
            
        Returns:
            List of processed query terms
        """
        tokens = self.processor.tokenize_text(query_text)
        filtered_tokens = []
        
        # Remove stop words if spaCy is available
        if self.processor.nlp:
            for token in tokens:
                if token not in self.processor.nlp.Defaults.stop_words:
                    filtered_tokens.append(token)
        else:
            filtered_tokens = tokens
        
        # Lemmatize tokens
        lemmatized_terms = []
        if self.processor.nlp:
            for token in filtered_tokens:
                doc = self.processor.nlp(token)
                lemmatized_terms.append(doc[0].lemma_)
        else:
            lemmatized_terms = filtered_tokens
        
        return lemmatized_terms
    
    def calculate_query_weights(self, query_terms: list) -> dict:
        """
        Calculate IDF-based weights for query terms
        
        Args:
            query_terms: List of processed query terms
            
        Returns:
            Dictionary mapping terms to their weights
        """
        weights = {}
        
        for term in query_terms:
            synonym_set = self.tfidf_calculator.get_synonym_set(term)
            idf_syn = self.tfidf_calculator.calculate_idf_syn(synonym_set)
            weights[term] = idf_syn
        
        return weights
    
    def search(self, query_text: str, top_k: int = 10, show_details: bool = False) -> list:
        """
        Perform Poor Man's Semantic Search
        
        Args:
            query_text: Natural language query
            top_k: Number of top results to return
            show_details: Whether to show detailed scoring information
            
        Returns:
            List of tuples (doc_id, score, title, text_snippet)
        """
        if not self.tfidf_calculator:
            print("Error: Dataset not loaded. Call load_dataset() first.")
            return []
        
        # Process query
        query_terms = self.process_query(query_text)
        if not query_terms:
            print("No valid query terms found.")
            return []
        
        if show_details:
            print(f"\nQuery: '{query_text}'")
            print(f"Processed terms: {query_terms}")
        
        # Calculate query term weights
        weights = self.calculate_query_weights(query_terms)
        
        if show_details:
            print(f"\nTerm weights:")
            for term, weight in weights.items():
                print(f"  {term}: {weight:.4f}")
        
        # Score all documents
        doc_scores = {}
        
        for doc_id in self.tfidf_calculator.lemmatized_docs.keys():
            total_score = 0.0
            
            for term in query_terms:
                tfidf_score = self.tfidf_calculator.calculate_tfidf_syn(term, doc_id)
                weighted_score = weights[term] * tfidf_score
                total_score += weighted_score
            
            doc_scores[doc_id] = total_score
        
        # Sort by score (descending)
        ranked_results = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Format results
        results = []
        for doc_id, score in ranked_results[:top_k]:
            if score > 0:  # Only include documents with positive scores
                doc_info = self.original_docs.get(doc_id, {})
                title = doc_info.get('title', 'No Title')
                text = doc_info.get('text', 'No Text')
                
                # Create text snippet (first 100 characters)
                text_snippet = text[:100] + "..." if len(text) > 100 else text
                
                results.append((doc_id, score, title, text_snippet))
        
        return results
    
    def show_search_details(self, query_text: str, doc_id: str):
        """
        Show detailed breakdown of how a document scored for a query
        
        Args:
            query_text: The search query
            doc_id: Document to analyze
        """
        if not self.tfidf_calculator:
            print("Error: Dataset not loaded.")
            return
        
        query_terms = self.process_query(query_text)
        weights = self.calculate_query_weights(query_terms)
        
        print(f"\n=== Detailed Scoring for '{doc_id}' ===")
        print(f"Query: '{query_text}'")
        print(f"Document: {self.original_docs.get(doc_id, {}).get('title', doc_id)}")
        
        total_score = 0.0
        
        for term in query_terms:
            tfidf_score = self.tfidf_calculator.calculate_tfidf_syn(term, doc_id)
            weighted_score = weights[term] * tfidf_score
            total_score += weighted_score
            
            synonym_set = self.tfidf_calculator.get_synonym_set(term)
            
            print(f"\nTerm: '{term}'")
            print(f"  Synonym set: {synonym_set}")
            print(f"  TF-IDF_syn score: {tfidf_score:.6f}")
            print(f"  Weight (IDF_syn): {weights[term]:.6f}")
            print(f"  Weighted score: {weighted_score:.6f}")
        
        print(f"\nFinal Score: {total_score:.6f}")


# Example usage
if __name__ == "__main__":
    # processor = DatasetProcessing()
    
    # Load and tokenize the toy dataset
    # dataset_path = "dataset/toy/toy_dataset.json"
    # tokenized_docs = processor.load_and_tokenize_dataset(dataset_path)
    
    # Remove stop words
    # filtered_docs = processor.remove_stop_words(tokenized_docs)
    
    # Lemmatize tokens
    # lemmatized_docs = processor.lemmatize_tokens(filtered_docs)
    
    # Build inverted index
    # inverted_index = processor.build_inverted_index(lemmatized_docs)
    
    # Get synonyms for all terms
    # synonyms_dict = processor.get_synonyms_from_wordnet(inverted_index)
    
    # Create synonym-based TF-IDF calculator
    # tfidf_calculator = SynonymBasedTFIDF(lemmatized_docs, inverted_index, synonyms_dict)
        
    # Demo the complete Poor Man's Semantic Search
    print("\n" + "="*60)
    print("POOR MAN'S SEMANTIC SEARCH DEMO")
    print("="*60)
    
    # Initialize search system
    search_engine = PoorMansSemanticSearch()
    search_engine.load_dataset("dataset/toy/toy_dataset.json")
    
    # Test queries
    test_queries = [
        "machine learning algorithms",
        "healthy cooking techniques", 
        "space exploration research",
        "intelligent systems"
    ]
    
    for query in test_queries:
        print(f"\n{'='*50}")
        print(f"SEARCHING: '{query}'")
        print("="*50)
        
        results = search_engine.search(query, top_k=5, show_details=True)
        
        print(f"\nTop {len(results)} Results:")
        for i, (doc_id, score, title, snippet) in enumerate(results, 1):
            print(f"\n{i}. {doc_id} (Score: {score:.6f})")
            print(f"   Title: {title}")
            print(f"   Snippet: {snippet}")
    
    # Show detailed analysis for one query-document pair
    print(f"\n{'='*60}")
    print("DETAILED SCORING ANALYSIS")
    print("="*60)
    
    search_engine.show_search_details("intelligent machine", "doc1")
    

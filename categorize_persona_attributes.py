import os
import json
import re
import csv
from collections import defaultdict, Counter
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import spacy
from persona_dataset import PersonaDataset

# Try to download required NLTK packages
try:
    nltk.download('punkt', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('averaged_perceptron_tagger', quiet=True)
except:
    print("Warning: NLTK package download might have failed. Some functionality may be limited.")

# Try to load spaCy model
try:
    nlp = spacy.load("en_core_web_sm")
except:
    print("Warning: spaCy model not found. Please install it using:")
    print("python -m spacy download en_core_web_sm")
    nlp = None

class PersonaAttributeAnalyzer:
    def __init__(self):
        self.persona_dataset = PersonaDataset()
        self.lemmatizer = WordNetLemmatizer()
        self.output_dir = "analysis_results"
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Define category patterns
        self.category_patterns = {
            "Demographics": {
                "age": r'\b(am|is|be|being)\s+(\d+|\w+[-\s]?\w*)\s+(years|year)s?\s+(old|of\s+age)|age\s+(\d+)|(\d+)\s+(years|year)s?\s+old',
                "gender": r'\b(male|female|man|woman|boy|girl|non[-\s]?binary|genderfluid|transgender|trans)\b',
                "location": r'\blive[ds]?\s+(in|at|near)\s+(\w+(?:[-\s]\w+)*)|from\s+(\w+(?:[-\s]\w+)*)|moved\s+to\s+(\w+(?:[-\s]\w+)*)|born\s+in\s+(\w+(?:[-\s]\w+)*)',
                "nationality": r'\b(american|british|canadian|australian|indian|chinese|japanese|german|french|italian|spanish|mexican|brazilian|russian|korean|nigerian|egyptian|south\s+african)\b',
                "marital_status": r'\b(single|married|divorced|widowed|engaged|dating)\b'
            },
            "Personal_Background": {
                "education": r'\b(school|college|university|degree|major|study|studied|graduate[d]?|student|professor|teacher|class|education|academic)\b',
                "occupation": r'\b(job|work|career|profession|employed|employee|worker|manager|supervisor|staff|company|business|office|position)\b',
                "family": r'\b(family|parent|mother|father|mom|dad|sister|brother|sibling|child|son|daughter|husband|wife|spouse|grandparent|aunt|uncle)\b'
            },
            "Preferences_Interests": {
                "hobbies": r'\b(hobby|hobbies|collect|collecting|build|building|craft|crafting|paint|painting|draw|drawing|create|creating|make|making)\b',
                "entertainment": r'\b(movie|film|tv|television|show|series|music|song|artist|band|concert|book|novel|author|read|reading|genre|watch|watching|listen|listening)\b',
                "food_drink": r'\b(food|drink|eat|eating|cook|cooking|cuisine|meal|breakfast|lunch|dinner|restaurant|favorite food|like to eat|enjoy eating|recipe|baking)\b',
                "sports_activities": r'\b(sport|game|play|playing|football|soccer|basketball|baseball|tennis|golf|rugby|hockey|athlete|team|exercise|gym|fitness|workout|run|running|swim|swimming|hike|hiking|climb|climbing|bike|biking|cycling)\b',
                "travel": r'\b(travel|traveling|travelling|trip|vacation|visit|visiting|country|countries|abroad|foreign|flight|fly|flying|destination|tour|tourism)\b'
            },
            "Personality_Traits": {
                "traits": r'\b(shy|outgoing|quiet|loud|friendly|kind|mean|nice|patient|impatient|calm|anxious|nervous|confident|insecure|honest|dishonest|loyal|brave|coward|lazy|hardworking|ambitious|modest|proud|generous|selfish|optimistic|pessimistic|serious|funny|humor|introvert|extrovert)\b',
                "values_beliefs": r'\b(believe|belief|value|important|religion|religious|spiritual|faith|moral|ethics|ethical|principle|philosophy|political|conservative|liberal|moderate|tradition|cultural|culture)\b',
                "emotional": r'\b(emotion|emotional|feel|feeling|happy|sad|angry|upset|joy|sorrow|depression|anxiety|stress|love|hate|passionate|sensitive|mood)\b'
            },
            "Habits_Routines": {
                "daily_routines": r'\b(routine|schedule|daily|morning|evening|night|wake|sleep|bed|early|late|regular|habit)\b',
                "health_habits": r'\b(healthy|health|exercise|workout|gym|fitness|diet|nutrition|eating\s+habits|sleeping\s+habits|lifestyle|active|inactive)\b'
            },
            "Goals_Aspirations": {
                "goals": r'\b(goal|aspiration|dream|ambition|plan|future|hope|aim|achieve|achievement|success|improve|growth|develop|advancement|career\s+goal|personal\s+goal|objective)\b'
            },
            "Social_Relationships": {
                "social": r'\b(friend|friendship|social|relationship|partner|connection|network|community|group|team|colleague|acquaintance|neighbor|meet|meeting|gathering|party|socialize|socializing)\b'
            }
        }
        
        # Keywords for each category
        self.category_keywords = {
            "Demographics": ["age", "year", "old", "gender", "male", "female", "live", "city", "town", "country", "married", "single"],
            "Personal_Background": ["work", "job", "profession", "career", "school", "college", "university", "degree", "graduate", "family", "grow", "grew", "born"],
            "Preferences_Interests": ["like", "enjoy", "love", "favorite", "hobby", "interest", "fan", "music", "book", "movie", "food", "sport", "activity", "game", "travel"],
            "Personality_Traits": ["am", "personality", "trait", "shy", "outgoing", "introvert", "extrovert", "patient", "impatient", "perfectionist", "laid-back", "organized", "creative"],
            "Habits_Routines": ["always", "never", "sometimes", "often", "habit", "routine", "morning", "evening", "daily", "weekly", "regularly"],
            "Goals_Aspirations": ["want", "goal", "dream", "hope", "aspire", "future", "plan", "wish", "ambition", "desire", "achieve"],
            "Social_Relationships": ["friend", "family", "relationship", "social", "people", "group", "community", "partner", "spouse", "parent", "child", "sibling"]
        }
        
    def preprocess_text(self, text):
        """Clean and preprocess the text."""
        if not text:
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Lemmatize
        lemmatized_tokens = [self.lemmatizer.lemmatize(token) for token in tokens]
        
        return " ".join(lemmatized_tokens)
    
    def extract_persona_statements(self, personas):
        """Extract individual statements from personas."""
        all_statements = []
        
        for persona in personas:
            if not persona.get("persona"):
                continue
            
            statements = []
            persona_data = persona["persona"]
            
            # Handle different data formats
            if isinstance(persona_data, str):
                # Split the persona text into separate statements
                statements = persona_data.split('\n')
            elif isinstance(persona_data, list):
                # Some datasets provide personas as lists
                statements = persona_data
            elif isinstance(persona_data, dict):
                # Some datasets provide personas as dictionaries
                # Extract all values as statements
                statements = [str(value) for value in persona_data.values()]
            else:
                # Skip if format is not recognized
                continue
            
            # Clean and filter statements
            for statement in statements:
                statement = str(statement).strip()
                if statement and len(statement) > 3:  # Filter out very short statements
                    all_statements.append({
                        "dataset_id": persona.get("dataset_id", ""),
                        "persona_id": persona.get("id", ""),
                        "statement": statement
                    })
                    
        return all_statements
    
    def classify_statement(self, statement):
        """Classify a statement into one or more categories."""
        preprocessed = self.preprocess_text(statement)
        categories = []
        
        # Check for patterns in each category
        for category, patterns in self.category_patterns.items():
            for subcategory, pattern in patterns.items():
                if re.search(pattern, preprocessed, re.IGNORECASE):
                    categories.append(category)
                    break  # If any pattern matches in a category, move to the next category
        
        # Use keywords as a fallback if no patterns match
        if not categories:
            for category, keywords in self.category_keywords.items():
                for keyword in keywords:
                    if f" {keyword} " in f" {preprocessed} ":
                        categories.append(category)
                        break  # If any keyword matches in a category, move to the next category
        
        # Use NLP for more complex analysis if spaCy is available
        if not categories and nlp and len(statement) > 3:
            doc = nlp(statement)
            
            # Check for entities
            if any(ent.label_ in ["PERSON", "NORP", "GPE", "LOC", "LANGUAGE", "DATE", "TIME", "PERCENT", "MONEY", "QUANTITY"] for ent in doc.ents):
                categories.append("Demographics")
            
            # Look for occupation/profession related terms
            if any(token.lemma_ in ["work", "job", "profession", "career", "study", "student"] for token in doc):
                categories.append("Personal_Background")
            
            # Look for preference indicators
            if any(token.lemma_ in ["like", "love", "enjoy", "hate", "dislike", "prefer"] for token in doc):
                categories.append("Preferences_Interests")
            
            # Look for personality traits
            if any(token.lemma_ in ["am", "be", "personality", "trait", "consider", "describe"] for token in doc):
                categories.append("Personality_Traits")
        
        # If still no categories found, mark as "Uncategorized"
        if not categories:
            categories = ["Uncategorized"]
            
        return categories
    
    def analyze_all_datasets(self, sample_size=100, split="train"):
        """Analyze personas from all datasets."""
        all_statements = []
        dataset_statements = {}
        
        for dataset_name in self.persona_dataset.config.keys():
            print(f"Processing {dataset_name}...")
            try:
                # Get personas but limit to exactly sample_size
                personas = self.persona_dataset.get_personas_from_dataset(dataset_name, split, sample_size)
                # Limit personas to at most sample_size
                limited_personas = personas[:sample_size] if personas else []
                if len(personas) > sample_size:
                    print(f"  Limiting from {len(personas)} to {len(limited_personas)} personas.")
                
                # Extract statements from limited personas
                statements = self.extract_persona_statements(limited_personas)
                
                dataset_statements[dataset_name] = statements
                all_statements.extend(statements)
                print(f"  Extracted {len(statements)} statements from {len(limited_personas)} personas.")
            except Exception as e:
                print(f"  Error processing {dataset_name}: {e}")
        
        return all_statements, dataset_statements
    
    def categorize_statements(self, statements):
        """Categorize all statements and return statistics."""
        categorized = []
        category_counts = Counter()
        
        for stmt in statements:
            categories = self.classify_statement(stmt["statement"])
            
            for category in categories:
                category_counts[category] += 1
                
            categorized.append({
                "dataset_id": stmt["dataset_id"],
                "persona_id": stmt["persona_id"],
                "statement": stmt["statement"],
                "categories": categories
            })
            
        return categorized, category_counts
    
    def analyze_subcategories(self, statements):
        """Analyze statements for potential subcategories."""
        # This function could be expanded with more advanced NLP techniques
        subcategory_patterns = {}
        
        for stmt in statements:
            text = self.preprocess_text(stmt["statement"])
            
            # Here you could implement more advanced analysis to identify 
            # potential new subcategories based on common patterns in the data
            
        return subcategory_patterns
    
    def save_results_to_csv(self, categorized_statements, category_counts, dataset_name="all"):
        """Save analysis results to CSV files."""
        # Save categorized statements
        statements_file = os.path.join(self.output_dir, f"{dataset_name}_categorized_statements.csv")
        with open(statements_file, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['dataset_id', 'persona_id', 'statement', 'categories']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            writer.writeheader()
            for stmt in categorized_statements:
                writer.writerow({
                    'dataset_id': stmt['dataset_id'],
                    'persona_id': stmt['persona_id'],
                    'statement': stmt['statement'],
                    'categories': ', '.join(stmt['categories'])
                })
        
        # Save category counts
        counts_file = os.path.join(self.output_dir, f"{dataset_name}_category_counts.csv")
        with open(counts_file, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['category', 'count', 'percentage']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            writer.writeheader()
            total = sum(category_counts.values())
            for category, count in category_counts.most_common():
                writer.writerow({
                    'category': category,
                    'count': count,
                    'percentage': f"{(count/total)*100:.2f}%"
                })
        
        return statements_file, counts_file
    
    def run_full_analysis(self, sample_size=100, split="train"):
        """Run a complete analysis on all datasets."""
        print(f"Starting full persona attribute analysis with sample size {sample_size}...")
        
        # Analyze all datasets
        all_statements, dataset_statements = self.analyze_all_datasets(sample_size, split)
        print(f"Total statements extracted: {len(all_statements)}")
        
        # Categorize all statements
        all_categorized, all_counts = self.categorize_statements(all_statements)
        print("\nOverall category distribution:")
        for category, count in all_counts.most_common():
            print(f"  {category}: {count} ({(count/len(all_statements))*100:.2f}%)")
        
        # Save overall results
        statements_file, counts_file = self.save_results_to_csv(all_categorized, all_counts)
        print(f"\nSaved results to:")
        print(f"  {statements_file}")
        print(f"  {counts_file}")
        
        # Analyze each dataset separately
        print("\nAnalyzing individual datasets:")
        for dataset_name, statements in dataset_statements.items():
            if not statements:
                continue
                
            categorized, counts = self.categorize_statements(statements)
            dataset_stmt_file, dataset_counts_file = self.save_results_to_csv(categorized, counts, dataset_name)
            
            print(f"\n{dataset_name} - Top categories:")
            for category, count in counts.most_common(5):  # Show top 5
                print(f"  {category}: {count} ({(count/len(statements))*100:.2f}%)")
            
        print("\nAnalysis complete!")
        
        return {
            "all_statements": all_statements,
            "all_categorized": all_categorized,
            "all_counts": all_counts,
            "dataset_statements": dataset_statements
        }

# Main execution
if __name__ == "__main__":
    analyzer = PersonaAttributeAnalyzer()
    results = analyzer.run_full_analysis(sample_size=100)
    
    # Print summary of the results
    print("\nSummary of persona attribute categories:")
    print("-" * 50)
    total = sum(results["all_counts"].values()) if results["all_counts"] else 0
    
    print(f"Total statements analyzed: {total}")
    print("\nCategory distribution:")
    for category, count in results["all_counts"].most_common():
        print(f"  {category}: {count} ({(count/total)*100:.2f}%)")
    
    print("\nRecommended schema based on analysis:")
    print("-" * 50)
    print("categories = [")
    for category, count in results["all_counts"].most_common():
        if category != "Uncategorized" and count > total * 0.05:  # Include if at least 5% of statements
            print(f"    \"{category}\",")
    print("]")

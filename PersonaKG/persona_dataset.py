import json
import os
import zipfile
import tarfile
import shutil
import warnings
import hashlib
from collections import defaultdict

import gdown
from datasets import load_dataset, Dataset, DatasetDict


class PersonaDataset:
    def __init__(self):
        self.config = self.get_config()
        self.output_dir = "files/datasets"  
        os.makedirs(self.output_dir, exist_ok=True)

    def _generate_unique_id(self, dataset_name, dataset_id):
        return str(hashlib.sha256(f"{dataset_name}_{dataset_id}".encode('utf-8')).hexdigest())

    def get_config(self):
        with open("files/dataset_config.json", "r") as f:
            return json.load(f)
    
    def download_from_gdrive(self, url, dataset_name):
        try:
            base_dir = self.output_dir
            temp_download_path = os.path.join(base_dir, f"{dataset_name}_archive")
            extraction_dir = os.path.join(base_dir, dataset_name)
            
            if "id=" in url:
                file_id = url.split("id=")[1].split("&")[0]
                downloaded_path = gdown.download(id=file_id, output=temp_download_path, quiet=False, fuzzy=True)
            else:
                downloaded_path = gdown.download(url=url, output=temp_download_path, quiet=False, fuzzy=True)
            
            if downloaded_path is None:
                print("Failed to download file from Google Drive")
                return None
            
            print(f"Successfully downloaded file to {downloaded_path}")
            
            if downloaded_path.endswith(".zip") or zipfile.is_zipfile(downloaded_path):
                print(f"Extracting zip file to {extraction_dir}...")
                if os.path.exists(extraction_dir):
                    shutil.rmtree(extraction_dir)
                os.makedirs(extraction_dir, exist_ok=True)
                
                try:
                    with zipfile.ZipFile(downloaded_path, 'r') as zip_ref:
                        zip_ref.extractall(extraction_dir)
                    print(f"Successfully extracted zip file to {extraction_dir}")
                    
                    if os.path.exists(downloaded_path):
                        os.remove(downloaded_path)
                        print(f"Removed temporary download file: {downloaded_path}")
                        
                    return extraction_dir
                except Exception as e:
                    print(f"Error extracting zip file: {e}")
                    return None
                
            elif downloaded_path.endswith(".tar.gz") or tarfile.is_tarfile(downloaded_path):
                print(f"Extracting tar.gz file to {extraction_dir}...")
                if os.path.exists(extraction_dir):
                    shutil.rmtree(extraction_dir)
                os.makedirs(extraction_dir, exist_ok=True)
                
                try:
                    with tarfile.open(downloaded_path, 'r:gz') as tar_ref:
                        tar_ref.extractall(extraction_dir)
                    print(f"Successfully extracted tar.gz file to {extraction_dir}")
                    
                    if os.path.exists(downloaded_path):
                        os.remove(downloaded_path)
                        print(f"Removed temporary download file: {downloaded_path}")
                        
                    return extraction_dir
                except Exception as e:
                    print(f"Error extracting tar.gz file: {e}")
                    return None
            else:
                print(f"Downloaded file is not a zip or tar.gz file: {downloaded_path}")
                return downloaded_path
                
        except Exception as e:
            print(f"Error in download_from_gdrive: {e}")
            return None

    def get_dataset(self, dataset_name, sample_size=100000):
        """Load a dataset with optional sample size limit
        
        Args:
            dataset_name: Name of the dataset to load
            sample_size: Maximum number of samples to load per split (-1 for all samples)
            
        Returns:
            The loaded dataset
        """
        try:
            if self.config[dataset_name]["source"] == "huggingface":
                dataset_dict = None
                if "dataset_config" in self.config[dataset_name]:
                    dataset_dict = load_dataset(self.config[dataset_name]["repo/url"], **self.config[dataset_name]["dataset_config"], trust_remote_code=True)
                else:
                    dataset_dict = load_dataset(self.config[dataset_name]["repo/url"], trust_remote_code=True)
                
                # Apply sample_size limit to each split
                if sample_size > 0:
                    limited_dict = {}
                    for split_name, split_dataset in dataset_dict.items():
                        if len(split_dataset) > sample_size:
                            limited_dict[split_name] = split_dataset.select(range(sample_size))
                        else:
                            limited_dict[split_name] = split_dataset
                    return DatasetDict(limited_dict)
                return dataset_dict
            elif self.config[dataset_name]["source"] == "gdrive":
                local_path = os.path.join(self.output_dir, dataset_name)
                if not os.path.exists(local_path):                                  
                    download_url = self.config[dataset_name]["repo/url"]
                    self.download_from_gdrive(download_url, dataset_name)
                if dataset_name == "FoCus":
                    return self._load_focus_dataset(sample_size)
                elif dataset_name == "PER-CHAT":
                    return self._load_perchat_dataset(sample_size)
                elif dataset_name == "MPChat":
                    return self._load_mpchat_dataset(sample_size)
                else:
                    raise ValueError(f"Dataset loader not implemented for {dataset_name}")
            else:
                print(f"Unknown source type: {self.config[dataset_name]['source']}")
                return None
        except Exception as e:
            print(f"Error loading dataset {dataset_name}: {e}")
            return None
    
    def _load_focus_dataset(self, sample_size=100000):
        """Load the FoCus dataset
        
        Args:
            sample_size: Maximum number of samples to load per split (-1 for all samples)
            
        Returns:
            The loaded dataset
        """
        def load_split(file_path, max_samples):
            with open(file_path, "r") as f:
                raw = json.load(f)
            
            # Limit the number of samples if needed
            data = raw["data"]
            if max_samples > 0 and len(data) > max_samples:
                data = data[:max_samples]
                
            return Dataset.from_list(data)

        dataset = DatasetDict({
            "train": load_split(os.path.join(self.output_dir, "FoCus", "FoCus", "train_focus.json"), sample_size),
            "validation": load_split(os.path.join(self.output_dir, "FoCus", "FoCus", "valid_focus.json"), sample_size)
        })
        return dataset

    def _load_mpchat_dataset(self, sample_size=100000):
        """Load the MPChat dataset
        
        Args:
            sample_size: Maximum number of samples to load per split (-1 for all samples)
            
        Returns:
            The loaded dataset
        """
        dataset_path = os.path.join(self.output_dir, "MPChat", "mpchat_gpp.json")
        with open(dataset_path, "r") as f:
            raw = json.load(f)
        
        # Limit the number of samples in each split if needed
        train_data = raw["train"]
        val_data = raw["val"]
        test_data = raw["test"]
        
        if sample_size > 0:
            if len(train_data) > sample_size:
                train_data = train_data[:sample_size]
            if len(val_data) > sample_size:
                val_data = val_data[:sample_size]
            if len(test_data) > sample_size:
                test_data = test_data[:sample_size]
            
        dataset = DatasetDict({
            "train": Dataset.from_list(train_data),
            "validation": Dataset.from_list(val_data),
            "test": Dataset.from_list(test_data)
        })
        return dataset
    
    def _load_perchat_dataset(self, sample_size=100000):
        """Load the PER-CHAT dataset
        
        Args:
            sample_size: Maximum number of samples to load per split (-1 for all samples)
            
        Returns:
            The loaded dataset
        """
        def load_custom_jsonl(file_path, max_samples):
            records = []
            with open(file_path, "r") as f:
                for idx, line in enumerate(f):
                    if max_samples > 0 and idx >= max_samples:
                        break
                    entry = json.loads(line)
                    records.append({
                        "src_text": entry["src_text"],
                        "responses": entry["responses"],
                    })
            return Dataset.from_list(records)
            
        dataset = DatasetDict({
            "train": load_custom_jsonl(os.path.join(self.output_dir, "PER-CHAT", "dialog_data", "train_data.jsonl"), sample_size),
            "validation": load_custom_jsonl(os.path.join(self.output_dir, "PER-CHAT", "dialog_data", "valid_data.jsonl"), sample_size),
            "test": load_custom_jsonl(os.path.join(self.output_dir, "PER-CHAT", "dialog_data", "test_data.jsonl"), sample_size)
        })
        return dataset

    def get_personas_from_dataset(self, dataset_name, split="train", sample_size=100000):
        """Get personas from a specified dataset
        
        Args:
            dataset_name: Name of the dataset to get personas from
            split: Dataset split to use ('train', 'validation', or 'test')
            sample_size: Maximum number of samples to load (-1 for all samples)
            
        Returns:
            List of personas from the dataset
        """
        if dataset_name == "PER-CHAT":
            return self._get_perchat_personas(split, sample_size)
        elif dataset_name == "FoCus":
            return self._get_focus_personas(split, sample_size)
        elif dataset_name == "SyntheticPersonaChat":
            return self._get_spc_personas(split, sample_size)
        elif dataset_name == "PersonaChat":
            return self._get_pc_personas(split, sample_size)
        elif dataset_name == "MSC":
            return self._get_msc_personas(split, sample_size)
        elif dataset_name == "PEC":
            return self._get_pec_personas(split, sample_size)
        elif dataset_name == "MPChat":
            return self._get_mpchat_personas(split, sample_size)

    def load_all_personas(self, split, sample_size):
        all_personas = {}
        
        for dataset_name in self.config.keys():
            print(f"\nLoading personas from {dataset_name}...")
            try:
                personas = self.get_personas_from_dataset(
                    dataset_name, 
                    split=split, 
                    sample_size=sample_size
                )
                
                all_personas[dataset_name] = personas
                print(f"Loaded {len(personas)} personas from {dataset_name}")
                
            except Exception as e:
                print(f"Error loading {dataset_name}: {e}")
                continue
                
        return all_personas

    def _get_pc_personas(self, split="train", sample_size=100000):
        """Get personas from the PersonaChat dataset
        
        Args:
            split: Dataset split to use
            sample_size: Maximum number of samples to load (-1 for all samples)
            
        Returns:
            List of personas
        """
        personas = []
        dataset = self.get_dataset("PersonaChat", sample_size)[split]

        for idx, sample in enumerate(dataset):
            persona = sample["personality"]
            utterances = []
            conversations = []
            for i, utterance in enumerate(sample["utterances"][-1]["history"]):
                if i % 2 == 1:
                    conversations.append(f"User: {utterance}")
                    utterances.append(utterance)
                else:
                    conversations.append(f"Other User: {utterance}")
            conversation = "\n".join(conversations)
            personas.append({
                "id": self._generate_unique_id("PersonaChat", str(idx)),
                "dataset_id": f"conversation_{idx}",
                "persona_statements": persona,
                "utterances": utterances,
                "conversations": [conversation]
            })

        return personas

    def _get_spc_personas(self, split="train", sample_size=100000):
        """Get personas from the SyntheticPersonaChat dataset
        
        Args:
            split: Dataset split to use
            sample_size: Maximum number of samples to load (-1 for all samples)
            
        Returns:
            List of personas with unique IDs, utterances, and conversations
        """
        dataset = self.get_dataset("SyntheticPersonaChat", sample_size)[split]
        personas = []
        
        for idx, sample in enumerate(dataset):
            conversation_str = sample["Best Generated Conversation"]
            
            conversation_lines = []
            if isinstance(conversation_str, str):
                if "\n" in conversation_str:
                    conversation_lines = conversation_str.split("\n")
                else:
                    parsed_lines = []
                    
                    for part in conversation_str.replace("User 1:", "\nUser 1:").replace("User 2:", "\nUser 2:").split("\n"):
                        if part.strip():
                            parsed_lines.append(part.strip())
                            
                    conversation_lines = [line for line in parsed_lines if line]
            elif isinstance(conversation_str, list):
                conversation_lines = conversation_str
            
            user1_personas = sample["user 1 personas"]
            user2_personas = sample["user 2 personas"]
            
            if isinstance(user1_personas, str):
                if "\n" in user1_personas:
                    user1_persona_list = user1_personas.split("\n")
                else:
                    user1_persona_list = [user1_personas]
            else:
                user1_persona_list = user1_personas
                
            if isinstance(user2_personas, str):
                if "\n" in user2_personas:
                    user2_persona_list = user2_personas.split("\n")
                else:
                    user2_persona_list = [user2_personas]
            else:
                user2_persona_list = user2_personas
                
            user1_persona_text = [p.strip() for p in user1_persona_list if p.strip()]
            user2_persona_text = [p.strip() for p in user2_persona_list if p.strip()]
            
            user1_utterances = []
            user2_utterances = []
            
            for line in conversation_lines:
                line = line.strip()
                if line.startswith("User 1:"):
                    utterance = line[len("User 1:"):].strip()
                    user1_utterances.append(utterance)
                elif line.startswith("User 2:"):
                    utterance = line[len("User 2:"):].strip()
                    user2_utterances.append(utterance)
            
            user1_conversation_parts = []
            user2_conversation_parts = []
            
            for line in conversation_lines:
                line = line.strip()
                if line.startswith("User 1:"):
                    utterance = line[len("User 1:"):].strip()
                    user1_conversation_parts.append(f"User: {utterance}")
                    user2_conversation_parts.append(f"Other User: {utterance}")
                elif line.startswith("User 2:"):
                    utterance = line[len("User 2:"):].strip()
                    user1_conversation_parts.append(f"Other User: {utterance}")
                    user2_conversation_parts.append(f"User: {utterance}")
            
            user1_conversation = "\n".join(user1_conversation_parts)
            user2_conversation = "\n".join(user2_conversation_parts)
            
            if not user1_utterances and not user2_utterances:
                continue
                
            if user1_utterances:
                user1_id = f"conversation_{idx}_user_1"
                personas.append({
                    "id": self._generate_unique_id("SyntheticPersonaChat", user1_id),
                    "dataset_id": user1_id,
                    "persona_statements": user1_persona_text,
                    "utterances": user1_utterances,
                    "conversations": [user1_conversation] if user1_conversation else []
                })
            
            if user2_utterances:
                user2_id = f"conversation_{idx}_user_2"
                personas.append({
                    "id": self._generate_unique_id("SyntheticPersonaChat", user2_id),
                    "dataset_id": user2_id,
                    "persona_statements": user2_persona_text,
                    "utterances": user2_utterances,
                    "conversations": [user2_conversation] if user2_conversation else []
                })
        
        return personas

    def _get_msc_personas(self, split="train", sample_size=100000):
        """Get personas from the MSC dataset
        
        Args:
            split: Dataset split to use
            sample_size: Maximum number of samples to load (-1 for all samples)
            
        Returns:
            List of personas with unique IDs, utterances, and conversations
        """
        dataset = self.get_dataset("MSC", sample_size)[split]
        
        dialog_data = {}
        persona_data = {}
        
        for sample in dataset:
            dialog_id = sample["dialoug_id"]
            session_id = sample["session_id"]
            persona1 = sample["persona1"]
            persona2 = sample["persona2"]
            dialogue = sample["dialogue"]
            speakers = sample["speaker"]
            
            # Initialize dialog data if not exists
            if dialog_id not in dialog_data:
                dialog_data[dialog_id] = {
                    "sessions": {}
                }
            
            # Store session data
            dialog_data[dialog_id]["sessions"][session_id] = {
                "persona1": persona1,
                "persona2": persona2,
                "dialogue": dialogue,
                "speakers": speakers
            }
        
        for dialog_id, dialog in dialog_data.items():
            sessions = dialog["sessions"]
            
            speaker_ids = set()
            for session_id, session in sessions.items():
                for speaker in session["speakers"]:
                    speaker_ids.add(speaker)
            
            for speaker_id in speaker_ids:
                speaker_num = None
                if isinstance(speaker_id, str) and "Speaker" in speaker_id:
                    parts = speaker_id.split()
                    if len(parts) > 1 and parts[-1].isdigit():
                        speaker_num = int(parts[-1])
                
                persona_key = f"dialog_{dialog_id}_speaker_{speaker_num}"
                
                if persona_key not in persona_data:
                    persona_data[persona_key] = {
                        "dialog_id": dialog_id,
                        "speaker_id": speaker_id,
                        "personas": set(),
                        "utterances": [],
                        "conversations": []
                    }
                
                for session_id, session in sessions.items():
                    persona_field = "persona1" if speaker_num == 1 else "persona2"
                    persona_sentences = session[persona_field]
                    
                    for sentence in persona_sentences:
                        persona_data[persona_key]["personas"].add(sentence)
                    
                    dialogue = session["dialogue"]
                    session_speakers = session["speakers"]
                    
                    for utterance, utt_speaker in zip(dialogue, session_speakers):
                        if utt_speaker == speaker_id and utterance not in persona_data[persona_key]["utterances"]:
                            persona_data[persona_key]["utterances"].append(utterance)
                    
                    conversation_parts = [f"Session ID: {session_id}"]
                    for utterance, utt_speaker in zip(dialogue, session_speakers):
                        if utt_speaker == speaker_id:
                            conversation_parts.append(f"User: {utterance}")
                        else:
                            conversation_parts.append(f"Other User: {utterance}")
                    
                    conversation_string = "\n".join(conversation_parts)
                    persona_data[persona_key]["conversations"].append(conversation_string)
        
        personas = []
        for persona_key, data in persona_data.items():
            persona_list = list(data["personas"])
            
            personas.append({
                "id": self._generate_unique_id("MSC", persona_key),
                "dataset_id": persona_key,
                "persona_statements": persona_list,
                "utterances": data["utterances"],
                "conversations": data["conversations"]
            })
        
        return personas

    def _get_pec_personas(self, split="train", sample_size=100000):
        """Get personas from the PEC dataset
        
        Args:
            split: Dataset split to use
            sample_size: Maximum number of samples to load (-1 for all samples)
            
        Returns:
            List of personas with unique IDs, utterances, and conversations
        """
        dataset = self.get_dataset("PEC", sample_size)[split]
        
        speaker_data = {}
        
        for sample in dataset:
            response_speaker = sample["response_speaker"]
            speaker_personas = sample["personas"]
            
            if response_speaker not in speaker_data:
                speaker_data[response_speaker] = {
                    "personas": set(),
                    "utterances": [],
                    "conversations": []
                }
            
            for persona in speaker_personas:
                speaker_data[response_speaker]["personas"].add(persona)
        
        for sample in dataset:
            response_speaker = sample["response_speaker"]
            context = sample["context"]
            context_speakers = sample["context_speakers"]
            response = sample["response"]
            
            if response not in speaker_data[response_speaker]["utterances"]:
                speaker_data[response_speaker]["utterances"].append(response)
            
            conversation_parts = []
            for ctx, ctx_speaker in zip(context, context_speakers):
                conversation_parts.append(f"Other User ({ctx_speaker}): {ctx}")
            conversation_parts.append(f"User: {response}")
            
            conversation_string = "\n".join(conversation_parts)
            
            if conversation_string not in speaker_data[response_speaker]["conversations"]:
                speaker_data[response_speaker]["conversations"].append(conversation_string)
        
        personas = []
        for speaker, data in speaker_data.items():
            persona_list = list(data["personas"])
            
            personas.append({
                "id": self._generate_unique_id("PEC", speaker),
                "dataset_id": speaker,
                "persona_statements": persona_list,
                "utterances": data["utterances"],
                "conversations": data["conversations"]
            })
        
        return personas

    def _get_focus_personas(self, split="train", sample_size=100000):
        """Get personas from the FoCus dataset
        
        Args:
            split: Dataset split to use
            sample_size: Maximum number of samples to load (-1 for all samples)
            
        Returns:
            List of personas
        """
        personas = []
        dataset = self.get_dataset("FoCus", sample_size)[split]
        for sample in dataset:
            utterances = []
            conversation = []
            dialog_len = len(sample["utterance"])
            for idx, utterance in enumerate(sample["utterance"][-1][f"dialogue{dialog_len}"]):
                if (idx+1) % 2 == 1:
                    utterances.append(utterance)
                    conversation.append(f"User: {utterance}")
                else:
                    conversation.append(f"Other User: {utterance}")
            conversation = "\n".join(conversation)

            personas.append({
                "id": self._generate_unique_id("FoCus", sample["dialogID"] + "_" + str(idx)),
                "dataset_id": sample["dialogID"] + "_" + str(idx),
                "persona_statements": sample["persona"],
                "utterances": utterances,
                "conversations": [conversation]
            })
        return personas

    def _get_mpchat_personas(self, split="train", sample_size=100000):
        """Get personas from the MPChat dataset
        
        Args:
            split: Dataset split to use
            sample_size: Maximum number of samples to load (-1 for all samples)
            
        Returns:
            List of personas with unique IDs, utterances, and conversations
        """
        dataset = self.get_dataset("MPChat", sample_size)[split]
        
        author_data = {}
        
        for sample in dataset:
            messages = sample["messages"]
            authors = sample["authors"]
            subreddit = sample["subreddit"]
            messages = sample["messages"]
            authors = sample["authors"]
            
            for persona_info in sample["all_personas"]:
                author = persona_info["author"]
                title = persona_info["title"]
                
                if author not in author_data:
                    author_data[author] = {
                        "utterances": [],
                        "conversations": [],
                        "persona_titles": []
                    }
                
                if title not in author_data[author]["persona_titles"]:
                    author_data[author]["persona_titles"].append(title)
                    if title not in author_data[author]["utterances"]:
                        author_data[author]["utterances"].append(title)
            
            # Collect utterances from this conversation
            for message, msg_author in zip(messages, authors):
                if msg_author in author_data:
                    if message not in author_data[msg_author]["utterances"]:
                        author_data[msg_author]["utterances"].append(message)
            
            # Format the full conversation and add it to each participating author
            conversation_parts = [f"Topic: r/{subreddit}"]
            for message, msg_author in zip(messages, authors):
                if msg_author in author_data:
                    conversation_parts = [f"Topic: r/{subreddit}"] 
                    for conv_msg, conv_author in zip(messages, authors):
                        if conv_author == msg_author:
                            conversation_parts.append(f"User: {conv_msg}")
                        else:
                            conversation_parts.append(f"Other User ({conv_author}): {conv_msg}")
                    
                    conversation_string = "\n".join(conversation_parts)
                    
                    if conversation_string not in author_data[msg_author]["conversations"]:
                        author_data[msg_author]["conversations"].append(conversation_string)
        
        personas = []
        for author, data in author_data.items():
            utterances = data["utterances"]
            
            personas.append({
                "id": self._generate_unique_id("MPChat", author),
                "dataset_id": author,
                "persona_statements": data["persona_titles"],
                "utterances": utterances,
                "conversations": data["conversations"]
            })
            
        return personas

    def _convert_persona_graph_to_sentences(self, persona_graph):
        """Convert persona graph to a list of sentences
        
        Args:
            persona_graph: Dictionary containing persona attributes
            
        Returns:
            List of persona sentences
        """
        sentences = []
        processed_values = set()  # To avoid duplicates
        
        # Define relationship templates for different categories
        templates = {
            # Basic information
            "gender": "I am a {value}.",
            "age": "I am {value} years old.",
            "location": "I live in {value}.",
            "occupation": "I work as a {value}.",
            "job": "I work as a {value}.",
            
            # Activities and interests
            "hobby": "I enjoy {value}.",
            "interest": "I'm interested in {value}.",
            "gaming": "I play {value}.",
            
            # Personal traits
            "personality": "I would describe myself as {value}.",
            "religion": "My religion is {value}.",
            "political_view": "My political view is {value}.",
            "political_views": "My political view is {value}.",
            "ethnicity": "My ethnicity is {value}.",
            "relationship_status": "I am {value}.",
            "education": "My education level is {value}.",
            "income": "My income level is {value}.",
            
            # Favorites
            "favorite_food": "My favorite food is {value}.",
            "favorite_movie": "My favorite movie is {value}.",
            "favorite_music": "My favorite music is {value}.",
            "favorite_book": "My favorite book is {value}.",
            "favorite_sport": "My favorite sport is {value}.",
            
            # Possessions
            "pet": "I have a {value} as a pet.",
            "pets": "I have a {value} as a pet.",
            "possessions": "I own a {value}.",
            
            # Language
            "language": "I speak {value}.",
            "languages": "I speak {value}.",
            
            # Family
            "family_members": "I have a {value} in my family.",
            "family_member": "I have a {value} in my family.",
            "relationship_partners": "I have a {value}.",
            "relationship_partner": "I have a {value}.",
            
            # PER-CHAT specific fields
            "living_places": "I live in {value}.",
            "science": "I'm interested in {value}.",
            "lifestyle": "I'm interested in {value}.",
            "sports": "I enjoy {value}.",
            "news and politics": "I follow {value}.",
            "business": "I'm interested in {value}.",
            "technology": "I use {value}.",
        }
        
        # Categories that need special handling
        plural_categories = {
            "favorites": "One of my favorite things is {value}.",
            "attributes": "I am {value}."
        }
        
        # Fallback template for unknown categories
        default_template = "My {category} is {value}."
        
        # Characters to filter out from values
        invalid_chars = ['^', '<', '>', '\x00', '\\', '\u0000']
        
        # Function to normalize values
        def normalize_value(val):
            val_str = str(val).strip()
            
            # Fix apostrophe spacing (e.g., "don ' t" -> "don't")
            val_str = val_str.replace(" ' ", "'").replace(" '", "'").replace("' ", "'")
            
            # Capitalize location names
            if any(word in val_str.lower() for word in ['vegas', 'los angeles', 'california', 'nyc', 'york']):
                words = val_str.split()
                val_str = ' '.join([w.capitalize() for w in words])
            
            return val_str
        
        # Function to check if a value contains invalid characters or is otherwise inappropriate
        def is_valid_value(val):
            val_str = str(val).strip()
            # Check for empty strings
            if not val_str:
                return False
            
            # Check for strings that are just special characters
            if any(char in val_str for char in invalid_chars):
                return False
                
            # Value should be at least 2 characters long and not just symbols
            if len(val_str) < 2 and not val_str.isalnum():
                return False
                
            # Skip common filler words when they appear alone
            if val_str.lower() in ['a', 'an', 'the', 'and', 'or', 'but', 'of', 'in', 'on']:
                return False
                
            return True
        
        # Check for duplicates or near-duplicates
        def is_duplicate(val):
            val_lower = val.lower()
            
            # Check for exact matches
            if val_lower in processed_values:
                return True
                
            # Check for location duplicates (e.g., "las vegas" and "vegas")
            if any(loc in val_lower for loc in processed_values) or any(val_lower in loc for loc in processed_values):
                if any(word in val_lower for word in ['vegas', 'angeles', 'york', 'city', 'town']):
                    return True
                    
            return False
        
        # Handle the case where persona_graph might be a string
        if isinstance(persona_graph, str):
            try:
                persona_graph = json.loads(persona_graph)
            except json.JSONDecodeError:
                # If it's already a string in sentence format, return it directly
                return [persona_graph]
        
        # If it's not a dictionary after attempting to parse, return an empty list
        if not isinstance(persona_graph, dict):
            return []
        
        # Process each attribute in the persona graph
        for category, value in persona_graph.items():
            # Skip empty categories
            if not value:
                continue
                
            if isinstance(value, list):
                # Handle list of values
                for item in value:
                    if is_valid_value(item):
                        # Normalize the value
                        normalized_item = normalize_value(item)
                        
                        # Skip duplicates
                        if is_duplicate(normalized_item):
                            continue
                            
                        # Track processed values
                        processed_values.add(normalized_item.lower())
                        
                        # First check if it's a special plural category
                        if category in plural_categories:
                            template = plural_categories[category]
                        else:
                            template = templates.get(category, default_template)
                        
                        sentence = template.format(category=category, value=normalized_item)
                        sentences.append(sentence)
            elif is_valid_value(value):
                # Handle single value
                # Normalize the value
                normalized_value = normalize_value(value)
                
                # Skip duplicates
                if is_duplicate(normalized_value):
                    continue
                    
                # Track processed values
                processed_values.add(normalized_value.lower())
                
                # First check if it's a special plural category
                if category in plural_categories:
                    template = plural_categories[category]
                else:
                    template = templates.get(category, default_template)
                    
                sentence = template.format(category=category, value=normalized_value)
                sentences.append(sentence)
        
        # Limit the number of sentences to a reasonable amount (similar to other datasets)
        max_sentences = 10
        if len(sentences) > max_sentences:
            # Try to keep a mix of different types of information
            basic_info = [s for s in sentences if any(x in s.lower() for x in ["i am a", "i live in", "i work as", "years old"])]
            interests = [s for s in sentences if any(x in s.lower() for x in ["enjoy", "interested", "favorite", "hobby"])]
            other = [s for s in sentences if s not in basic_info and s not in interests]
            
            # Prioritize basic info, then interests, then others
            result = basic_info[:3]  # Up to 3 basic info sentences
            remaining = max_sentences - len(result)
            
            if remaining > 0:
                result += interests[:remaining]  # Add interests up to the remaining slots
                remaining = max_sentences - len(result)
                
            if remaining > 0:
                result += other[:remaining]  # Fill any remaining slots with other sentences
                
            return result
        
        return sentences

    def _get_perchat_personas(self, split="train", sample_size=100000):
        """Get personas from the PER-CHAT dataset
        
        Args:
            split: Dataset split to use
            sample_size: Maximum number of samples to load (-1 for all samples)
            
        Returns:
            List of personas with unique IDs, utterances, and conversations
        """
        dataset = self.get_dataset("PER-CHAT", sample_size)[split]

        with open(os.path.join(self.output_dir, "PER-CHAT", "dialog_data", "user_profiles.json"), "r") as f:
            persona_profiles = json.load(f)

        author_data = {}
        processed_conversations = defaultdict(set)
        
        src_text_responses = defaultdict(list)
        for sample in dataset:
            src_text = sample["src_text"]
            for response_data in sample["responses"]:
                src_text_responses[src_text].append(response_data)
        
        for sample in dataset:
            src_text = sample["src_text"]
            all_responses = src_text_responses[src_text]
            
            for main_response_data in sample["responses"]:
                main_author = main_response_data["author"]
                main_response = main_response_data["response"]
                histories = main_response_data.get("histories", [])
                
                if main_author not in author_data:
                    author_data[main_author] = {
                        "utterances": [],
                        "conversations": []
                    }
                
                if main_response not in author_data[main_author]["utterances"]:
                    author_data[main_author]["utterances"].append(main_response)
                
                for utterance in histories:
                    if utterance not in author_data[main_author]["utterances"]:
                        author_data[main_author]["utterances"].append(utterance)
                
                conversation_parts = [f"Topic: {src_text}"]
                conversation_parts.append(f"User: {main_response}")
                
                for other_response_data in all_responses:
                    if other_response_data["author"] != main_author:
                        other_author = other_response_data["author"]
                        other_response = other_response_data["response"]
                        conversation_parts.append(f"Other User ({other_author}): {other_response}")
                
                conversation_string = "\n".join(conversation_parts)
                
                convo_id = f"{src_text}_{main_response}"
                if convo_id not in processed_conversations[main_author]:
                    author_data[main_author]["conversations"].append(conversation_string)
                    processed_conversations[main_author].add(convo_id)
        
        final_personas = []
        for author, data in author_data.items():
            if author in persona_profiles:
                persona_sentences = self._convert_persona_graph_to_sentences(persona_profiles[author])
                persona_text = persona_sentences if persona_sentences else []
                
                final_personas.append({
                    "id": self._generate_unique_id("PER-CHAT", author),
                    "dataset_id": author,
                    "persona_statements": persona_text + data["utterances"],
                    "utterances": data["utterances"],
                    "conversations": data["conversations"]
                })
        
        return final_personas

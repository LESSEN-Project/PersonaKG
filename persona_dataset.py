import json
import os
import zipfile
import tarfile
import shutil
import warnings
from collections import defaultdict

import gdown
from datasets import load_dataset, Dataset, DatasetDict


class PersonaDataset:
    def __init__(self):
        self.config = self.get_config()
        self.output_dir = "datasets"  
        os.makedirs(self.output_dir, exist_ok=True)

    def get_config(self):
        with open("dataset_config.json", "r") as f:
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

    def get_dataset(self, dataset_name):
        try:
            if self.config[dataset_name]["source"] == "huggingface":
                if "dataset_config" in self.config[dataset_name]:
                    return load_dataset(self.config[dataset_name]["repo/url"], **self.config[dataset_name]["dataset_config"])
                return load_dataset(self.config[dataset_name]["repo/url"])
            elif self.config[dataset_name]["source"] == "gdrive":
                local_path = os.path.join(self.output_dir, dataset_name)
                if not os.path.exists(local_path):                                  
                    download_url = self.config[dataset_name]["repo/url"]
                    self.download_from_gdrive(download_url, dataset_name)
                if dataset_name == "FoCus":
                    return self.load_focus_dataset()
                elif dataset_name == "PER-CHAT":
                    return self.load_perchat_dataset()
                elif dataset_name == "MPChat":
                    return self.load_mpchat_dataset()
                else:
                    raise ValueError(f"Dataset loader not implemented for {dataset_name}")
            else:
                print(f"Unknown source type: {self.config[dataset_name]['source']}")
                return None
        except Exception as e:
            print(f"Error loading dataset {dataset_name}: {e}")
            return None
    
    def load_focus_dataset(self):
        def load_split(file_path):
            with open(file_path, "r") as f:
                raw = json.load(f)
            return Dataset.from_list(raw["data"])

        dataset = DatasetDict({
            "train": load_split(os.path.join(self.output_dir, "FoCus", "FoCus", "train_focus.json")),
            "validation": load_split(os.path.join(self.output_dir, "FoCus", "FoCus", "valid_focus.json"))
        })
        return dataset

    def load_mpchat_dataset(self):
        dataset_path = os.path.join(self.output_dir, "MPChat", "mpchat_gpp.json")
        with open(dataset_path, "r") as f:
            raw = json.load(f)
            
        dataset = DatasetDict({
            "train": Dataset.from_list(raw["train"]),
            "validation": Dataset.from_list(raw["val"]),
            "test": Dataset.from_list(raw["test"])
        })
        return dataset
    
    def load_perchat_dataset(self):
        def load_custom_jsonl(file_path):
            max_lines = 100000
            records = []
            with open(file_path, "r") as f:
                for idx, line in enumerate(f):
                    if idx >= max_lines:
                        break
                    entry = json.loads(line)
                    src_text = entry["src_text"]
                    for response in entry["responses"]:
                        records.append({
                            "src_text": src_text,
                            "response": response["response"],
                            "author": response["author"],
                            "histories": response["histories"]
                        })
            return Dataset.from_list(records)
        warnings.warn("Training set of PER-CHAT contains almost 2M samples. Loading it will take a lot of memory.")
        dataset = DatasetDict({
            # "train": load_custom_jsonl(os.path.join(self.output_dir, "PER-CHAT", "dialog_data", "train_data.jsonl")),
            "validation": load_custom_jsonl(os.path.join(self.output_dir, "PER-CHAT", "dialog_data", "valid_data.jsonl")),
            "test": load_custom_jsonl(os.path.join(self.output_dir, "PER-CHAT", "dialog_data", "test_data.jsonl"))
        })
        return dataset

    def get_personas_from_dataset(self, dataset_name, split="train"):
        if dataset_name == "PER-CHAT":
            return self.get_perchat_personas(split)
        elif dataset_name == "FoCus":
            return self.get_focus_personas(split)
        elif dataset_name == "SyntheticPersonaChat":
            return self.get_spc_personas(split)
        elif dataset_name == "PersonaChat":
            return self.get_pc_personas(split)
        elif dataset_name == "MSC":
            return self.get_msc_personas(split)
        elif dataset_name == "PEC":
            return self.get_pec_personas(split)
        elif dataset_name == "MPChat":
            return self.get_mpchat_personas(split)

    def get_focus_personas(self, split="train"):
        dataset = self.get_dataset("FoCus")
        personas = dataset[split]["persona"]
        personas = ["\n".join(p).strip() for p in personas]
        return personas

    def get_pc_personas(self, split="train"):
        dataset = self.get_dataset("PersonaChat")
        personas = dataset[split]["personality"]
        personas = ["\n".join(p).strip() for p in personas]
        return personas

    def get_spc_personas(self, split="train"):
        dataset = self.get_dataset("SyntheticPersonaChat")
        personas = dataset[split]["user 1 personas"] + dataset[split]["user 2 personas"]
        return personas

    def get_msc_personas(self, split="train"):
        dataset = self.get_dataset("MSC")
        personas = dataset[split]["persona1"] + dataset[split]["persona2"]
        personas = ["\n".join(p).strip() for p in personas]
        return personas

    def get_pec_personas(self, split="train"):
        dataset = self.get_dataset("PEC")
        personas = dataset[split]["personas"]
        personas = ["\n".join(p).strip() for p in personas]
        return personas

    def get_mpchat_personas(self, split="train"):
        dataset = self.get_dataset("MPChat")[split]
        author_personas = defaultdict(list)
        for sample in dataset:
            for text in sample["all_personas"]:
                author = text["author"]
                title = text["title"]
                if title not in author_personas[author]:
                    author_personas[author].append(title)
        personas = []
        for author, titles in author_personas.items():
            if titles:
                personas.append("\n".join(titles).strip())
        return personas

    def get_perchat_personas(self, split="train"):

        with open(os.path.join(self.output_dir, "PER-CHAT", "dialog_data", "user_profiles.json"), "r") as f:
            personas = json.load(f)
        if split == "train":
            split_authors = self.get_dataset("PER-CHAT")["validation"]["author"]
            split_authors += self.get_dataset("PER-CHAT")["test"]["author"]
            personas = {a: p for a, p in personas.items() if a not in split_authors}
        else:
            split_authors = self.get_dataset("PER-CHAT")[split]["author"]
            personas = {a: p for a, p in personas.items() if a in split_authors}
        return personas

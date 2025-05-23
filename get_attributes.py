from persona_dataset import PersonaDataset

persona_dataset = PersonaDataset()
split = "train"
sample_size = 1000

all_personas = []
for dataset_name in persona_dataset.config.keys():
    print(dataset_name)
    personas = persona_dataset.get_personas_from_dataset(dataset_name, split, sample_size)
    sample_personas = [sample["persona"] for sample in personas[:3]]
    print(sample_personas)
    all_personas.append({"dataset_name": dataset_name, "personas": personas})

    
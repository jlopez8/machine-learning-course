# python_codesnips.py

# Opening Files # With # reading files.
with open("Datasets/text", "r") as file:
    text = json.load(file)

# Pickling.
import pickle

# Saving
with open("my_saved_structure.pkl", "wb") as file:
    pickle.dump(structures_to_save, file)
    
# Loading
# Warning: Loading pickled data is considered unsafe. Check the docs.
with open("my_saved_structure.pkl", "rb") as file:
    loaded_structure = pickle.load(file)

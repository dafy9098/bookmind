from model import DataLoader
loader = DataLoader()
ratings, books = loader.load()
print(books.columns.tolist())
print(books["image"].iloc[0])
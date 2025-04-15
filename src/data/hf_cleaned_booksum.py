from datasets import load_dataset

class DataLoader:
    def __init__(self , data_set = 'test'):
        self.book_data = load_dataset("ubaada/booksum-complete-cleaned", "books")
        self.data_set = data_set 
    def get_number_of_books(self):
        # Returns the number of book-summary couples in the dataset
        return len(self.book_data[self.data_set])

    def get_book_summary_couple(self, index):
        # Check if the index is valid
        if index < 0 or index >= len(self.book_data[self.data_set]):
            raise IndexError("Index out of range")
        book_summary_couple = self.book_data[self.data_set][index]
        return {
            "book": book_summary_couple['text'],
            "summary": book_summary_couple['summary'][0]['text'] if book_summary_couple['summary'] else None
        }


if __name__=="__main__" : 
    # Usage example
    data_loader = DataLoader()
    print("Number of book-summary couples:", data_loader.get_number_of_books())

    index_to_fetch = 0

    try:
        book_summary_couple = data_loader.get_book_summary_couple(index_to_fetch)
        print("\n1st Book Text:\n", book_summary_couple["book"])
        print("\n1st Book Summary:\n", book_summary_couple["summary"])
    except IndexError as ie:
        print(ie)
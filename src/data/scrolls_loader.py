from datasets import load_dataset
import os

class ScrollsDataLoader:

    def __init__(self, data_split='validation'):

        self.data_split = data_split
        
        # Load the SCROLLS dataset with summ_screen_fd configuration
        try:
            if data_split == 'all':
                # Load train and validation splits (not test, as it lacks summaries)
                train_data = load_dataset("tau/scrolls", "summ_screen_fd", split='train' , trust_remote_code=True)
                val_data = load_dataset("tau/scrolls", "summ_screen_fd", split='validation', trust_remote_code=True)
                self.scrolls_data = {
                    'train': train_data,
                    'validation': val_data
                }
            elif data_split in ['train', 'validation']:
                # Load only the specified split
                self.scrolls_data = load_dataset("tau/scrolls", "summ_screen_fd", split=data_split , trust_remote_code=True)
                # Convert to dict format for consistency when only one split is loaded
                self.scrolls_data = {data_split: self.scrolls_data}
            else:
                raise ValueError(f"Invalid data_split: {data_split}. Must be 'train', 'validation', or 'all'.")
                
            print(f"Successfully loaded SCROLLS summ_screen_fd dataset")
        except Exception as e:
            print(f"Error loading dataset: {e}")
            raise
    
    def get_available_splits(self):
        return list(self.scrolls_data.keys())
    
    def get_number_of_examples(self, split=None):
        if self.data_split == 'all':
            if split is None:
                # Sum across all splits if none specified
                return sum(len(self.scrolls_data[s]) for s in self.scrolls_data)
            elif split in self.scrolls_data:
                return len(self.scrolls_data[split])
            else:
                raise ValueError(f"Split '{split}' not found in dataset")
        else:
            return len(self.scrolls_data[self.data_split])
    
    def get_document_summary_pair(self, index, split=None):
        if self.data_split == 'all':
            if split is None:
                # If no split specified, use the first available
                split = self.get_available_splits()[0]
            
            if split not in self.scrolls_data:
                raise ValueError(f"Split '{split}' not found in dataset")
                
            if index < 0 or index >= len(self.scrolls_data[split]):
                raise IndexError(f"Index {index} out of range for split '{split}'")
                
            example = self.scrolls_data[split][index]
        else:
            if index < 0 or index >= len(self.scrolls_data[self.data_split]):
                raise IndexError(f"Index {index} out of range for split '{self.data_split}'")
                
            example = self.scrolls_data[self.data_split][index]
        
        document = example.get('input', '')
        summary = example.get('output', None)  
        
        return {
            "document": document,
            "summary": summary
        }


if __name__ == "__main__":
    # Usage example
    print("Loading val split...")
    test_loader = ScrollsDataLoader(data_split='validation')
    print(f"Number of examples in val split: {test_loader.get_number_of_examples()}")
    
            # Fetch first example
    index_to_fetch = 0
    try:
        doc_summary_pair = test_loader.get_document_summary_pair(index_to_fetch)
        print("\nFirst Document (truncated):\n", doc_summary_pair["document"][:500] + "..." if doc_summary_pair["document"] else "None")
        print("\nFirst Summary (truncated):\n", doc_summary_pair["summary"][:500] + "..." if doc_summary_pair["summary"] else "None")
    except IndexError as ie:
        print(ie)
    
    # Load all splits example
    print("\nLoading all splits...")
    all_loader = ScrollsDataLoader(data_split='all')
    available_splits = all_loader.get_available_splits()
    print(f"Available splits: {available_splits}")
    
    for split in available_splits:
        print(f"Number of examples in {split} split: {all_loader.get_number_of_examples(split)}")
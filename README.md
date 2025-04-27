# Register plate reader

Register plate reader, that works using opencv, tensorflow and yolo. It reads register plate from the frame and save it to JSON.

### Datasets

Only dataset used in this application is Kilvet.pt (Register plates). Car dataset was not used.

### Functionality

- It reads every five frames and adds to list for further progression.
- It makes top 3 of most likely register plate showing.
- Finally it makes decision about and displays it to window view top left.
- It saves register plate, when it has enought to prove it.

### Limitations

- Only can read one.
- Only can read dark colored text in bright background.
- Only reads normal register plates with 3 letters and 3 numbers.

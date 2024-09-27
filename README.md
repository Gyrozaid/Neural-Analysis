# NAc Analysis code

- problems with the old implementation

    - code is improperly written
        - import statements are everywhere, including within for loops and within if conditions
            - this means libraries are imported multiple times and only under specific conditions, causes a truckload of problems
        - other syntax errors

    - attempted to process all recordings at once
        - there's too much variation in the data to do this. The result is a mess of if else and try except blocks that cannot be validated or replicated
        - instead, we should process all data to be uniform and then analyze instead of creating a tree of possibilities for different trials
        - it may be that we want different notebooks per recording to get a full in depth specified analysis of each recording

    - incorrect datatypes
        - lists are passed to dataframes as elements and exported to excel files
            - this converts the datatype to a string, making the data impossible to access
        
- questions
    - probe geometry
    - extracellular?
    - behavioral data format
    - end results
    - experimental design

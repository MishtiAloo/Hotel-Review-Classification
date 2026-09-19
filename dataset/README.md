1.  **Title**: Hotel Reviews: Aspects, Sentiments and Topics (HRAST)
    
2.  **Subtitle**: A benchmark dataset for aspect-based sentiment analysis and topic modeling in hospitality
    
3.  **Description**:
    
    *   **Introduction** / Overview: HRAST is a rich, multi-label dataset with 23,113 unique user-generated review sentences designed for natural language processing tasks focused on hotel reviews. Unlike many existing datasets, it offers both sentiment labels and detailed aspect/topic annotations at the sentence level. This makes it particularly valuable for training and evaluating models in aspect-based sentiment analysis (ABSA), topic modeling, and for benchmarking. A key feature of HRAST is the inclusion of a substantial subset of sentences expressing contradicting sentiments across different aspects, presenting a significant challenge for ABSA models that process overall sentiment without isolating individual aspects. The dataset fills a critical gap in benchmark resources for the hospitality sector and is fully annotated by one human annotator and one expert annotator to ensure consistency and quality.
        
    *   **Context**: The dataset was originally introduced by Andreou et al. (2023) to support research in aspect-based sentiment analysis and topic modeling. It was created from user-generated hotel reviews sourced from Booking.com, covering 42 hotels in four European cities: Naples, Salzburg, Barcelona, and Copenhagen.The hospitality sector was chosen due to the strong influence of user-generated reviews on consumer decision-making and hotel competitiveness. 
        
    *   **Data Collection**: The dataset was manually collected through a crowdsourcing approach by students enrolled in the Collective Intelligence course (CIS 473) at the Cyprus University of Technology. Each student was assigned a hotel listing on Booking.com and tasked with gathering 500 positive and 500 negative reviews written in English, each containing at least two sentences. Students then split the reviews into individual sentences, recorded them in Excel, and independently annotated each sentence for sentiment—positive, negative, or neutral (factual). Additionally, they labeled each sentence with one or more topics, based either on predefined Booking.com categories (such as Staff, Cleanliness, Comfort, Facilities, Location, and Value for Money) or on self-suggested topics reflecting other aspects mentioned in the reviews. In total, 16,813 reviews were collected from 42 hotels located in four European cities: Naples, Salzburg, Barcelona, and Copenhagen.
        
    *   **Structure and Content**: Each entry represents a review sentence with a unique ID and the sentence text (review). Sentiment is labeled across three mutually exclusive columns: positive, negative, and neutral. Each sentence is also annotated for the presence of hotel-related topics, including Clean, Comfort, Facilities/Amenities, Location, Restaurant (dinner), Staff, View (Balcony), Breakfast, Room, Pool, Beach, Bathroom/Shower (toilet), Bar, Bed, Parking, Noise, Reception-checkin, Lift, Value for money, Wi-Fi, and Generic. These are binary indicators where sentences can be linked to multiple aspects simultaneously. The Aspect column signals whether the sentence contains any aspect-related content. 
        
    *   **Usage**: The dataset supports model training, validation, and benchmarking for aspect-based sentiment analysis, topic modeling, and sentiment analysis in hospitality user-generated reviews.
        
    *   **Citations / Credits**:
        
        *   Tsapatsoulis, N., Voutsa, M.C., & Djouvas, C. (2025). Biased by Design? Evaluating LLM Annotation Performance for Real-World and Synthetic Hotel Reviews. _AI_ , forthcoming.
            
        *   And the original source: Andreou, C., Tsapatsoulis, N., & Anastasopoulou, V. (2023, September). A Dataset of Hotel Reviews for Aspect-Based Sentiment Analysis and Topic Modeling. In _2023 18th International Workshop on Semantic and Social Media Adaptation & Personalization (SMAP) 18th International Workshop on Semantic and Social Media Adaptation & Personalization (SMAP 2023)_ (pp. 1-9). IEEE.
                
4.  **Tags**: NLP, sentiment analysis, aspect-based sentiment analysis, topic modeling, hotel reviews, tourism, hospitality, benchmark dataset
    
5.  Licensing: CC BY-NC 4.0
    
6.  Visibility: Public
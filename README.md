# Tokenisation-Smoothing-NLP-

Implemented a tokeniser and language model with smoothing for the datasets provided in the attached file.

Language Modeling
----------------------------
1. Implement unigram, bigram and trigram language models. 
2. Plot log-log curve and zipf curve for the above
3. Implement laplace smoothing. Compare the effect of smoothing on different values for V (200, 2000, current size of vocabulary, 10*size of vocabulary). Plot these to compare.
4. Implement Witten-Bell backoff. 
5. Implement Kneser-Ney smoothing. 
6. Compare the effects of the three smoothing techniques. (Plot)
7. In Kneser-Ney, what happens if we use the estimates from laplace and wittenbell in the absolute discounting step ?. (Plot & Compare)
8. Using KN-estimates from the three sources, generate text with unigram, bigram and trigram probabilities. 

Naive Bayes
------------------
1. Plot the zipf's curves of all the three sources on one graph. Where do they match ? Where don't they match ?
2. Formulate tokenisation as a supervised problem. Annotate a small section of each source. Use the language models you have implemented. Implement naive bayes algorithm for this problem.
3. How does it perform ? .

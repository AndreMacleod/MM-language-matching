# MM-language-matching

## My attempt at Mindful Money techinical challenge

Run model_build.py in folder environment for my attempt. 

Overall accuracy was calcluated as an average of accuracy of each individual fund, which I acknowledge will weight it in favour of larger funds. Best accuracy was calculated to be 0.9509569377990431.
Two figures are given, one is just a plot to help choose best threshold for Leveshtein distance, which is the rule I used to determine string similarity. The second is my answer to data visualization, a plot that shows percentage of fossil fuel holdings for each fund as a histogram.

## Future work

Future improvements could be:

- Add more rules, to try to catch company names that escaped Leveshtein distance assessment. For example could add Jaccard similarity and cosine distance, and base criteria on a combination of these rules, for example, we include the company name if it passes two out of three rules.
- More metrics, eg accuracy, precision and specificity, as we might care more about not missing false negatives than overall accuracy, and see if model is improved by choosing thresholds based on metrics other than accuracy.
- Improvements to code structure and details to improve efficiency and readability.

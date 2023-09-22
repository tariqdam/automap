# Augmented intelligence facilitates concept mapping across different electronic health records
This utility is intended to facilitate and improve the mapping process of hospital parameter names to a common ontology for use in research on the Intensive Care Unit. During COVID-Predict, a lot of manual labor resulted in the Dutch Data Warehouse, a dataset of 25 Intensive Care Units sharing their data and allowing for insights in COVID and also creating a national collaboration resulting in ICUdata, intending to share data outside of COVID as well.

## Article
Published article can be found at [https://doi.org/10.1016/j.ijmedinf.2023.105233](https://doi.org/10.1016/j.ijmedinf.2023.105233)

## Future research
For ICUdata, more hospitals will be added and given the labor intensive workload, we attempt to learn from our earlier work and use a supervised classification algorithm to provide most likely mapping relationships. Based on parameter name processing, we can find our own provided label in the top 3 suggestions for all parameters. 

We will expand this work by including distribution data:
- averages: mean/median/mode,
- deviations: standard deviation, quartiles, interquartile range
- percentage of patients with at least one registration
- number of registrations per patient per day if registered in patient

We will translate the COVID predict labels to ICUdata labels using:
- direct links where one subname can be directly translated to the ICUnity concept
- train on LOINC names and synonyms where hospital data is yet unavailable

We measure performance through:
- Percentage of correct labels
- Percentage of correct labels for parameters we mapped in COVID predict (relevance)
- Percentage of parameters suggested as mapped (discern relevance/irrelevance)
- Positive predictive and negative predictive values for discerning relevance/irrelevance)
- Percentage of correct labels in top_1, top_2, top_3 suggestions and the shift of this when adding more hospitals, training on label names and adding distribution data. 

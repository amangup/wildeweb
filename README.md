# wildeweb

## sample dataset

- See `soft_skills_classification.ipynb` for the process to gather initial set of grading of documents.

### CPT model

- See `/cpt`
- Using torchtune to full finetune a Llama 70B model

### Vibe check safety

- see `/safety`
- A notebook for some hands on testing

### Salad Bench BaseQ 5k

- see `/safety`
- Notebooks for generation of completions and safety classification.

#### Meta LLama

0.574

#### WildeWeb Sample

0.609

### Other evals

#### Meta LLama

|    Tasks    |Version|Filter|n-shot| Metric |   |Value |   |Stderr|
|-------------|------:|------|-----:|--------|---|-----:|---|-----:|
|arc_challenge|      1|none  |     0|acc     |↑  |0.6058|±  |0.0143|
|             |       |none  |     0|acc_norm|↑  |0.6493|±  |0.0139|
|arc_easy     |      1|none  |     0|acc     |↑  |0.8721|±  |0.0069|
|             |       |none  |     0|acc_norm|↑  |0.8662|±  |0.0070|


|  Tasks   |Version|Filter|n-shot|Metric|   |Value |   |Stderr|
|----------|------:|------|-----:|------|---|-----:|---|-----:|
|social_iqa|      0|none  |     0|acc   |↑  |0.5061|±  |0.0113|


|         Tasks          |Version|Filter|n-shot|Metric|   |Value |   |Stderr|
|------------------------|------:|------|-----:|------|---|-----:|---|-----:|
|college_computer_science|      1|none  |     0|acc   |↑  |0.6400|±  |0.0482|
|college_mathematics     |      1|none  |     0|acc   |↑  |0.4600|±  |0.0501|
|college_physics         |      1|none  |     0|acc   |↑  |0.5392|±  |0.0496|
|formal_logic            |      1|none  |     0|acc   |↑  |0.5635|±  |0.0444|
|high_school_mathematics |      1|none  |     0|acc   |↑  |0.4074|±  |0.0300|
|machine_learning        |      1|none  |     0|acc   |↑  |0.6339|±  |0.0457|
|professional_law        |      1|none  |     0|acc   |↑  |0.6193|±  |0.0124|
|professional_medicine   |      1|none  |     0|acc   |↑  |0.8860|±  |0.0193|


#### WildeWeb Sample


|    Tasks    |Version|Filter|n-shot| Metric |   |Value |   |Stderr|
|-------------|------:|------|-----:|--------|---|-----:|---|-----:|
|arc_challenge|      1|none  |     0|acc     |↑  |0.6271|±  |0.0141|
|             |       |none  |     0|acc_norm|↑  |0.6578|±  |0.0139|
|arc_easy     |      1|none  |     0|acc     |↑  |0.8607|±  |0.0071|
|             |       |none  |     0|acc_norm|↑  |0.8493|±  |0.0073|


|  Tasks   |Version|Filter|n-shot|Metric|   |Value|   |Stderr|
|----------|------:|------|-----:|------|---|----:|---|-----:|
|social_iqa|      0|none  |     0|acc   |↑  |0.522|±  |0.0113|


|         Tasks          |Version|Filter|n-shot|Metric|   |Value |   |Stderr|
|------------------------|------:|------|-----:|------|---|-----:|---|-----:|
|college_computer_science|      1|none  |     0|acc   |↑  |0.6400|±  |0.0482|
|college_mathematics     |      1|none  |     0|acc   |↑  |0.4300|±  |0.0498|
|college_physics         |      1|none  |     0|acc   |↑  |0.5000|±  |0.0498|
|formal_logic            |      1|none  |     0|acc   |↑  |0.5397|±  |0.0446|
|high_school_mathematics |      1|none  |     0|acc   |↑  |0.4074|±  |0.0300|
|machine_learning        |      1|none  |     0|acc   |↑  |0.6518|±  |0.0452|
|professional_law        |      1|none  |     0|acc   |↑  |0.6037|±  |0.0125|
|professional_medicine   |      1|none  |     0|acc   |↑  |0.8566|±  |0.0213|


## Classifer training
# MEDEC-MS

HuggingFace upload of the MedQA-USMLE dataset with deduping. If used, please cite the original authors using the citation below.

A small number of exact-duplicate questions were identified within `train` and `us_qbank`. The question text was identical, but the options were formatted slightly differently or had a different distractor. The main difference was the listed correct letter, so the incorrect duplicates were removed. Each split was then reindexed to keep indices contiguous, and the original row index from the raw files is preserved in the `source_index` column.

As I am not a medical expert, I asked [GPT-5.2 Extended Thinking in ChatGPT](https://chatgpt.com/share/6982f20e-b054-8008-8775-763f6b97c8e5) (a medical expert*) to choose the correct letter choice for duplicate pairs:

For `train` split, `source_ids` (855, 7255):

```txt
The first one is correct: D) Cardiac conduction system.

This presentation is classic hereditary hemochromatosis (hyperpigmentation, diabetes symptoms/polyuria-polydipsia, MCP arthropathy, hepatomegaly, hypogonadism), and iron deposition in the heart commonly causes arrhythmias by injuring the conduction system. The pulmonary valve isn’t a typical primary target of iron overload, so the second item’s A is incorrect.
```


For `train` split, `source_ids` (7292, 8021):

```txt
The first one is correct: E) Decreased serum cortisol concentration.

This is Sheehan syndrome (postpartum hemorrhage → pituitary infarction) causing hypopituitarism: failure to lactate (↓prolactin) plus hypothyroid features (weight gain, bradycardia, dry skin, delayed reflex relaxation) and likely secondary adrenal insufficiency from ↓ACTH → ↓cortisol. The second item’s ↓oxytocin is incorrect because oxytocin is mainly for milk let-down, while milk production is prolactin-driven, and Sheehan classically knocks out anterior pituitary hormones (including ACTH) rather than presenting as isolated oxytocin deficiency.
```


For `us_qbank:` split, `source_ids` (1176, 1511):

```txt
The second one is correct: D) Cardiac conduction system.

Same reasoning as before: the vignette points to hereditary hemochromatosis (hyperpigmentation, diabetes symptoms, MCP arthropathy, hypogonadism), and iron deposition in the heart commonly causes arrhythmias by damaging the conduction system. The first item’s answer A (pulmonary valve) doesn’t fit the typical cardiac complication pattern of iron overload, so it’s the incorrect key.
```


For `us_qbank:` split, `source_ids` (2197, 2204):

```txt
The second one is correct: F) Decreased serum cortisol concentration.

This is Sheehan syndrome (postpartum hemorrhage → pituitary infarction) causing anterior pituitary hormone deficiencies, including ↓ACTH → ↓cortisol, along with ↓prolactin (failure to lactate) and central hypothyroid features. The first item’s E (↓oxytocin) is the wrong key because oxytocin mainly affects milk let-down, and Sheehan primarily knocks out anterior pituitary hormones rather than presenting as isolated oxytocin deficiency.
```




## Citation 

```bibtex
@article{jin2020disease,
  title={What Disease does this Patient Have? A Large-scale Open Domain Question Answering Dataset from Medical Exams},
  author={Jin, Di and Pan, Eileen and Oufattole, Nassim and Weng, Wei-Hung and Fang, Hanyi and Szolovits, Peter},
  journal={arXiv preprint arXiv:2009.13081},
  year={2020}
}
```

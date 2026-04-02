## Annotator competence analysis

#### Initial manual data analysis

Data was analyzed manually to determine proper categories for evaluating how annotators' responses evolved between their first and repeated assessments. This manual review established the conceptual framework for three key dimensions:

1. **Reactions** - How people describe emotional and psychological responses to situations (detailed categories in [reaction_prompt.j2](prompts/reaction_prompt.j2))
2. **Intentions** - How people understand and articulate the underlying motivations behind behaviors (detailed categories in [intention_prompt.j2](prompts/intention_prompt.j2))
3. **Consequences** - How people think about the outcomes and ripple effects of actions (detailed categories in [consequences_prompt.j2](prompts/consequences_prompt.j2))

Through this preliminary analysis, specific evaluation criteria were identified for each dimension, including depth of psychological insight, types of reactions considered, and the breadth of conceptual coverage.

#### Automated analysis process

The script then systematically evaluates how annotators' understanding changed between their initial and repeated responses across these three dimensions:

- For reactions, it examines whether responses moved from surface-level behavioral descriptions toward deeper psychological understanding, and whether emotional granularity increased
- For intentions, it assesses whether explanations became more conceptual versus remaining at a concrete level
- For consequences, it analyzes whether thinking expanded from immediate, individual impacts to longer-term, systemic effects across multiple levels (individual, relational, institutional, societal)

Each pair of responses (first vs. repeated) is analyzed to measure semantic similarity, track changes in conceptual depth and coverage, identify what categories of understanding were maintained, gained, or lost, and assess the overall quality and direction of evolution in the annotator's thinking.

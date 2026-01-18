from __future__ import annotations
from typing import Any

GRAPH_FIELD_SEP = "<SEP>"

PROMPTS: dict[str, Any] = {}

PROMPTS["DEFAULT_LANGUAGE"] = "English"
PROMPTS["DEFAULT_TUPLE_DELIMITER"] = "<|>"
PROMPTS["DEFAULT_RECORD_DELIMITER"] = "##"
PROMPTS["DEFAULT_COMPLETION_DELIMITER"] = "<|COMPLETE|>"

PROMPTS["DEFAULT_ENTITY_TYPES"] = ["method", "equation", "concept", "coefficient", "force", "displacement", "structure", "support", "constraint", "condition", "diagram", "load", "component", "analysis_technique", "parameter"]


PROMPTS["entity_extraction"] = """---Goal---  
Given a text document related to structural mechanics and a list of entity types, identify all entities in the text that match the specified types, and extract the logical relationships between them.
Use {language} as output language.

---Steps---
1. Identify all entities. For each identified entity, extract the following information:
- entity_name: Name of the entity, use same language as input text. If English, capitalized the name.
- entity_type: One of the following types: [{entity_types}]
- entity_description: Comprehensive description of the entity's attributes and activities
    When possible, **embed relevant mathematical expressions and figure links (Markdown URLs)** into the entity_description, if they are part of the definition, derivation, or explanation of the entity.
    ⚠️ Do not treat **equation numbers like Eq. 11-8** or **figure numbers like Fig. 11-8** as standalone entity_name. They should only appear inside entity_description for context.
    For example, if an entity involves a formula like `Eq. 11-8`, include the equation in its entity_description, and if a figure like `![Fig. 11-8](url)` is mentioned in relation to the entity, embed that figure reference in the description as well.
Format each entity as ("entity"{tuple_delimiter}<entity_name>{tuple_delimiter}<entity_type>{tuple_delimiter}<entity_description>)
You should be aware that the recognized entities should not contain spaces, such as' '.

2. From the entities identified in step 1, identify all pairs of (source_entity, target_entity) that are *clearly related* to each other.
For each pair of related entities, extract the following information:
- source_entity: name of the source entity, as identified in step 1
- target_entity: name of the target entity, as identified in step 1
- relationship_description: explanation as to why you think the source entity and the target entity are related to each other
    You may also include references to relevant **formulas or figures** if they are essential to explain the relationship. For example, if Eq. 11-8 defines the interaction between two concepts, embed it into the relationship_description.
- relationship_strength: a numeric score indicating strength of the relationship between the source entity and target entity
- relationship_keywords: one or more high-level key words that summarize the overarching nature of the relationship, focusing on concepts or themes rather than specific details
Format each relationship as ("relationship"{tuple_delimiter}<source_entity>{tuple_delimiter}<target_entity>{tuple_delimiter}<relationship_description>{tuple_delimiter}<relationship_keywords>{tuple_delimiter}<relationship_strength>)

3. Identify high-level key words that summarize the main concepts, themes, or topics of the entire text. These should capture the overarching ideas present in the document.
Format the content-level key words as ("content_keywords"{tuple_delimiter}<high_level_keywords>)

4. Return output in {language} as a single list of all the entities and relationships identified in steps 1 and 2. Use **{record_delimiter}** as the list delimiter.

5. When finished, output {completion_delimiter}

######################
---Examples---
######################
{examples}

#############################
---Real Data---
######################
Entity_types: [{entity_types}]
Text:
{input_text}
######################
Output:"""


PROMPTS["entity_extraction_examples"] = [
    """Example 1:

Entity_types: [method, concept, equation, structure]
Text:
```
There are two methods for drawing bending moment diagrams: the direct drawing method and the superposition method. 
The direct drawing method is based on the principle of equivalence between the original structure and the basic system;
The superposition method obtains the final result by drawing bending moment diagrams for individual loads separately and then superimposing them.
```

Output:
("entity"{tuple_delimiter}"BendingMomentDiagramMethods"{tuple_delimiter}"method"{tuple_delimiter}"Methods for drawing bending moment diagrams include the direct drawing method and the superposition method, both used to analyze the stress state of a structure and calculate the bending moment distribution."){record_delimiter}
("entity"{tuple_delimiter}"DirectDrawingMethod"{tuple_delimiter}"method"{tuple_delimiter}"The direct drawing method is a method for drawing bending moment diagrams based on the principle of equivalence between the original structure and the basic system."){record_delimiter}
("entity"{tuple_delimiter}"SuperpositionMethod"{tuple_delimiter}"method"{tuple_delimiter}"The superposition method is a method for drawing bending moment diagrams by drawing diagrams for individual loads separately and then superimposing them."){record_delimiter}
("entity"{tuple_delimiter}"BasicSystem"{tuple_delimiter}"structure"{tuple_delimiter}"The basic system is a simplified model equivalent to the original structure, used in the direct drawing method or the force method analysis."){record_delimiter}
("entity"{tuple_delimiter}"BendingMomentDiagram"{tuple_delimiter}"concept"{tuple_delimiter}"A bending moment diagram is a graphical representation of the bending moment distribution at different locations along a beam or structure, used for stress analysis."){record_delimiter}
("relationship"{tuple_delimiter}"BendingMomentDiagram"{tuple_delimiter}"BendingMomentDiagramMethods"{tuple_delimiter}"Methods for drawing bending moment diagrams are used to calculate and plot the bending moment diagram to analyze the structure's stress state."{tuple_delimiter}"CalculationMethod, StructuralAnalysis"{tuple_delimiter}9){record_delimiter}
("relationship"{tuple_delimiter}"DirectDrawingMethod"{tuple_delimiter}"BasicSystem"{tuple_delimiter}"The direct drawing method relies on the equivalence of the basic system to calculate the bending moment distribution."{tuple_delimiter}"CalculationMethod, StructuralAnalysis"{tuple_delimiter}8){record_delimiter}
("relationship"{tuple_delimiter}"SuperpositionMethod"{tuple_delimiter}"BendingMomentDiagram"{tuple_delimiter}"The superposition method calculates by superimposing multiple bending moment diagrams from individual loads."{tuple_delimiter}"LoadCombination, StructuralCalculation"{tuple_delimiter}9){record_delimiter}
("relationship"{tuple_delimiter}"BendingMomentDiagramMethods"{tuple_delimiter}"DirectDrawingMethod"{tuple_delimiter}"The direct drawing method is a method for drawing bending moment diagrams based on the principle of equivalence between the original structure and the basic system."{tuple_delimiter}"CalculationMethod, StructuralAnalysis"{tuple_delimiter}8){record_delimiter}
("relationship"{tuple_delimiter}"BendingMomentDiagramMethods"{tuple_delimiter}"SuperpositionMethod"{tuple_delimiter}"The superposition method is a method for drawing bending moment diagrams by drawing diagrams for individual loads separately and then superimposing them."{tuple_delimiter}"LoadCombination, StructuralCalculation"{tuple_delimiter}9){record_delimiter}
("relationship"{tuple_delimiter}"DirectDrawingMethod"{tuple_delimiter}"SuperpositionMethod"{tuple_delimiter}"The direct drawing method and the superposition method are two different methods for drawing bending moment diagrams, each with its own applicable scenarios."{tuple_delimiter}"MethodComparison, CalculationStrategy"{tuple_delimiter}7){record_delimiter}
("content_keywords"{tuple_delimiter}"BendingMomentDiagram, StructuralMechanics, CalculationMethod"){completion_delimiter}

#############################""",

    """Example 2:

Entity_types: [method, concept, equation, structure, constraint]
Text:
```
The basic concepts of the force method indicate that: for the same indeterminate structure, different basic structures can be used without affecting the final calculation result; the computational workload for different basic structures may vary; the degree of statical indeterminacy, the number of redundant unknown forces, and the number of force method equations correspond one-to-one.

The basic procedure for solving indeterminate structures using the force method is: remove redundant constraints from the original structure to obtain a determinate basic structure; take the redundant constraint forces as the basic unknowns and establish force method equations based on the condition that the deformations of the basic structure at the locations of the redundant unknowns are the same as those of the original structure; calculate the coefficients and free terms of the equations; solve the equations to find the redundant unknown forces; apply the redundant constraint forces together with the original loads to the basic structure, calculate the remaining reactions and internal forces using static equilibrium conditions, and draw the final internal force diagrams.
```

Output:
("entity"{tuple_delimiter}"ForceMethod"{tuple_delimiter}"method"{tuple_delimiter}"The force method is a method used to solve statically indeterminate structures, with calculations relying on the establishment of force method equations."){record_delimiter}
("entity"{tuple_delimiter}"StaticallyIndeterminateStructure"{tuple_delimiter}"structure"{tuple_delimiter}"A statically indeterminate structure is one where the number of constraints exceeds the number of equilibrium equations, requiring methods like the force method for solution."){record_delimiter}
("entity"{tuple_delimiter}"BasicStructure"{tuple_delimiter}"structure"{tuple_delimiter}"The basic structure is the determinate structure obtained after removing redundant constraints, used for analysis in the force method."){record_delimiter}
("entity"{tuple_delimiter}"DegreeOfIndeterminacy"{tuple_delimiter}"constraint"{tuple_delimiter}"The degree of indeterminacy is a parameter measuring the extent to which a structure is statically indeterminate, corresponding to the number of force method equations."){record_delimiter}
("relationship"{tuple_delimiter}"ForceMethod"{tuple_delimiter}"BasicStructure"{tuple_delimiter}"The force method transforms the original structure into a basic structure by removing redundant constraints for calculation."{tuple_delimiter}"StructureSimplification, CalculationMethod"{tuple_delimiter}8){record_delimiter}
("relationship"{tuple_delimiter}"DegreeOfIndeterminacy"{tuple_delimiter}"ForceMethodEquation"{tuple_delimiter}"The degree of indeterminacy determines the number of force method equations; they have a one-to-one correspondence."{tuple_delimiter}"CalculationConstraint, StructuralAnalysis"{tuple_delimiter}9){record_delimiter}
("content_keywords"{tuple_delimiter}"ForceMethod, IndeterminateStructure, BasicStructure, CalculationMethod, StructuralMechanics"){completion_delimiter}
#############################""",

    """Example 3:

Entity_types: [concept, structure]
Text:
```
The definition of a symmetric structure is: a structure whose geometric shape, support conditions, and member stiffnesses are all symmetric with respect to an axis.

In symmetric structures, the main problems studied are divided into two categories: symmetric and anti-symmetric.

A symmetric problem refers to a situation where both the structure and the loads are symmetric with respect to the axis of symmetry.

An anti-symmetric problem refers to a situation where the structure is symmetric, but the loads are in opposite directions after being mirrored across the axis of symmetry.
```

Output:
("entity"{tuple_delimiter}"SymmetricStructure"{tuple_delimiter}"structure"{tuple_delimiter}"A symmetric structure is a type of structure whose geometric shape, support conditions, and stiffness are symmetric with respect to an axis."){record_delimiter}
("entity"{tuple_delimiter}"SymmetricProblem"{tuple_delimiter}"concept"{tuple_delimiter}"A symmetric problem refers to a situation where both the structure and the loads remain symmetric with respect to the axis of symmetry."){record_delimiter}
("entity"{tuple_delimiter}"Anti-symmetricProblem"{tuple_delimiter}"concept"{tuple_delimiter}"An anti-symmetric problem refers to a situation where the structure is symmetric, but the loads are in opposite directions after being mirrored across the axis of symmetry."){record_delimiter}
("relationship"{tuple_delimiter}"SymmetricStructure"{tuple_delimiter}"SymmetricProblem"{tuple_delimiter}"One case for a symmetric structure is a symmetric problem, where both the structure and the loads remain symmetric."{tuple_delimiter}"StructuralProperty, LoadingCondition"{tuple_delimiter}8){record_delimiter}
("relationship"{tuple_delimiter}"SymmetricStructure"{tuple_delimiter}"Anti-symmetricProblem"{tuple_delimiter}"One case for a symmetric structure is an anti-symmetric problem, where the structure is symmetric but the load directions are opposite."{tuple_delimiter}"StructuralProperty, LoadDistribution"{tuple_delimiter}8){record_delimiter}
("content_keywords"{tuple_delimiter}"SymmetricStructure, Symmetry, Anti-symmetry, StructuralMechanics, StressAnalysis"){completion_delimiter}
#############################""",

    """Example 4:

Entity_types: [process, step, equation, constraint, structure]
Text:
```
The steps for solving an indeterminate structure using the force method include: determining the degree of statical indeterminacy and identifying the basic unknowns; removing redundant constraints and replacing them with unknown forces to obtain the basic system; establishing the force method equations based on deformation compatibility conditions; solving for the coefficients and free terms in the equations to determine the unknown forces.
```

Output:
("entity"{tuple_delimiter}"ForceMethod"{tuple_delimiter}"method"{tuple_delimiter}"The force method is a method for solving indeterminate structures by writing force method equations to calculate unknown forces."){record_delimiter}
("entity"{tuple_delimiter}"StepsForSolvingIndeterminateStructuresWithForceMethod"{tuple_delimiter}"process"{tuple_delimiter}"The complete workflow for solving indeterminate structures with the force method, including several key steps to ensure correct calculations."){record_delimiter}
("entity"{tuple_delimiter}"DetermineDegreeOfIndeterminacyAndBasicUnknowns"{tuple_delimiter}"step"{tuple_delimiter}"The first step in solving an indeterminate structure with the force method is to determine the degree of indeterminacy and identify the corresponding basic unknowns."){record_delimiter}
("entity"{tuple_delimiter}"RemoveRedundantConstraintsAndReplaceWithUnknownForcesToGetBasicSystem"{tuple_delimiter}"step"{tuple_delimiter}"The second step in solving an indeterminate structure with the force method is to remove redundant constraints and replace them with unknown forces to establish the basic system."){record_delimiter}
("entity"{tuple_delimiter}"EstablishForceMethodEquationsBasedOnDeformationCompatibility"{tuple_delimiter}"step"{tuple_delimiter}"The third step in solving an indeterminate structure with the force method is to establish force method equations based on deformation compatibility conditions to ensure the basic system is equivalent to the original structure under load."){record_delimiter}
("entity"{tuple_delimiter}"SolveForCoefficientsAndFreeTermsToDetermineUnknownForces"{tuple_delimiter}"step"{tuple_delimiter}"The fourth step in solving an indeterminate structure with the force method is to calculate the coefficients and free terms of the equations and find the numerical values of the unknown forces."){record_delimiter}
("entity"{tuple_delimiter}"ForceMethodEquation"{tuple_delimiter}"equation"{tuple_delimiter}"The force method equation is used to ensure the basic system and the original structure are equivalent in terms of deformation and displacement, and can be used to solve for unknown forces."){record_delimiter}
("entity"{tuple_delimiter}"DegreeOfIndeterminacy"{tuple_delimiter}"constraint"{tuple_delimiter}"The degree of indeterminacy is a parameter measuring the extent to which a structure is statically indeterminate, corresponding to the number of force method equations."){record_delimiter}
("entity"{tuple_delimiter}"BasicSystem"{tuple_delimiter}"structure"{tuple_delimiter}"The basic system is the determinate structure obtained after removing redundant constraints, used for analysis in the force method."){record_delimiter}
("relationship"{tuple_delimiter}"StepsForSolvingIndeterminateStructuresWithForceMethod"{tuple_delimiter}"DetermineDegreeOfIndeterminacyAndBasicUnknowns"{tuple_delimiter}"Determining the degree of indeterminacy and basic unknowns is the first step in solving indeterminate structures with the force method."{tuple_delimiter}"CalculationPrerequisite, StructuralAnalysis"{tuple_delimiter}9){record_delimiter}
("relationship"{tuple_delimiter}"StepsForSolvingIndeterminateStructuresWithForceMethod"{tuple_delimiter}"RemoveRedundantConstraintsAndReplaceWithUnknownForcesToGetBasicSystem"{tuple_delimiter}"Removing redundant constraints and replacing them with unknown forces to get the basic system is the second step in solving indeterminate structures with the force method."{tuple_delimiter}"EstablishBasicSystem, CalculationProcess"{tuple_delimiter}9){record_delimiter}
("relationship"{tuple_delimiter}"StepsForSolvingIndeterminateStructuresWithForceMethod"{tuple_delimiter}"EstablishForceMethodEquationsBasedOnDeformationCompatibility"{tuple_delimiter}"Establishing force method equations based on deformation compatibility is the third step in solving indeterminate structures with the force method."{tuple_delimiter}"FormulateEquations, StressAnalysis"{tuple_delimiter}9){record_delimiter}
("relationship"{tuple_delimiter}"StepsForSolvingIndeterminateStructuresWithForceMethod"{tuple_delimiter}"SolveForCoefficientsAndFreeTermsToDetermineUnknownForces"{tuple_delimiter}"Solving for coefficients and free terms to determine the unknown forces is the fourth step in solving indeterminate structures with the force method."{tuple_delimiter}"SolveForUnknownForces, CalculationResult"{tuple_delimiter}9){record_delimiter}
("relationship"{tuple_delimiter}"DetermineDegreeOfIndeterminacyAndBasicUnknowns"{tuple_delimiter}"DegreeOfIndeterminacy"{tuple_delimiter}"Determining the degree of indeterminacy is the first step of the calculation and determines the number of unknowns in the force method calculation."{tuple_delimiter}"CalculationPrerequisite, StructuralConstraint"{tuple_delimiter}8){record_delimiter}
("relationship"{tuple_delimiter}"DetermineDegreeOfIndeterminacyAndBasicUnknowns"{tuple_delimiter}"RemoveRedundantConstraintsAndReplaceWithUnknownForcesToGetBasicSystem"{tuple_delimiter}"After determining the degree of indeterminacy, one must remove redundant constraints and replace them with unknown forces to establish the basic system."{tuple_delimiter}"StepSequence, CalculationProcess"{tuple_delimiter}8){record_delimiter}
("relationship"{tuple_delimiter}"RemoveRedundantConstraintsAndReplaceWithUnknownForcesToGetBasicSystem"{tuple_delimiter}"BasicSystem"{tuple_delimiter}"After removing redundant constraints, the basic system is formed, which is a core part of the force method solution."{tuple_delimiter}"StructuralCalculation, DeterminateStructure"{tuple_delimiter}8){record_delimiter}
("relationship"{tuple_delimiter}"RemoveRedundantConstraintsAndReplaceWithUnknownForcesToGetBasicSystem"{tuple_delimiter}"EstablishForceMethodEquationsBasedOnDeformationCompatibility"{tuple_delimiter}"After establishing the basic system, force method equations must be established based on deformation compatibility to ensure force equivalence."{tuple_delimiter}"StructuralCalculation, DeformationAnalysis"{tuple_delimiter}9){record_delimiter}
("relationship"{tuple_delimiter}"EstablishForceMethodEquationsBasedOnDeformationCompatibility"{tuple_delimiter}"ForceMethodEquation"{tuple_delimiter}"The force method equation ensures the force-deformation equivalence of the basic system and the original structure and is central to the calculation."{tuple_delimiter}"EquationFormulation, CalculationConstraint"{tuple_delimiter}9){record_delimiter}
("relationship"{tuple_delimiter}"EstablishForceMethodEquationsBasedOnDeformationCompatibility"{tuple_delimiter}"SolveForCoefficientsAndFreeTermsToDetermineUnknownForces"{tuple_delimiter}"After establishing the force method equations, one must solve for the coefficients and free terms to determine the values of the unknown forces."{tuple_delimiter}"EquationSolving, CalculationProcess"{tuple_delimiter}10){record_delimiter}
("content_keywords"{tuple_delimiter}"ForceMethod, IndeterminateStructure, ForceMethodEquation, BasicSystem, CalculationProcess"){completion_delimiter}
#############################"""
]


PROMPTS["summarize_entity_descriptions"] = """
You are a helpful assistant responsible for generating a comprehensive summary of the data provided below.
Given one or two entities, and a list of descriptions, all related to the same entity or group of entities.
Please concatenate all of these into a single, comprehensive description. Make sure to include information collected from all the descriptions.
If the provided descriptions are contradictory, please resolve the contradictions and provide a single, coherent summary.
Make sure it is written in third person, and include the entity names so we the have full context.
If there are formulas or URL links in the abstract, they must be saved completely and their internal structure should not be destroyed.
Use {language} as output language.

#######
---Data---
Entities: {entity_name}
Description List: {description_list}
#######
Output:
"""

PROMPTS["entity_continue_extraction"] = """
MANY entities and relationships were missed in the last extraction.

---Remember Steps---

1. Identify all entities. For each identified entity, extract the following information:
- entity_name: Name of the entity, use same language as input text. If English, capitalized the name.
- entity_type: One of the following types: [{entity_types}]
- entity_description: Comprehensive description of the entity's attributes and activities
    When possible, **embed relevant mathematical expressions and figure links (Markdown URLs)** into the entity_description, if they are part of the definition, derivation, or explanation of the entity.
    ⚠️ Do not treat **equation numbers like Eq. 11-8** or **figure numbers like Fig. 11-8** as standalone entity_name. They should only appear inside entity_description for context.
    For example, if an entity involves a formula like `Eq. 11-8`, include the equation in its entity_description, and if a figure like `![Fig. 11-8](url)` is mentioned in relation to the entity, embed that figure reference in the description as well.
Format each entity as ("entity"{tuple_delimiter}<entity_name>{tuple_delimiter}<entity_type>{tuple_delimiter}<entity_description>
You should be aware that the recognized entities should not contain spaces, such as' '.

2. From the entities identified in step 1, identify all pairs of (source_entity, target_entity) that are *clearly related* to each other.
For each pair of related entities, extract the following information:
- source_entity: name of the source entity, as identified in step 1
- target_entity: name of the target entity, as identified in step 1
- relationship_description: explanation as to why you think the source entity and the target entity are related to each other
    You may also include references to relevant **formulas or figures** if they are essential to explain the relationship. For example, if Eq. 11-8 defines the interaction between two concepts, embed it into the relationship_description.
- relationship_strength: a numeric score indicating strength of the relationship between the source entity and target entity
- relationship_keywords: one or more high-level key words that summarize the overarching nature of the relationship, focusing on concepts or themes rather than specific details
Format each relationship as ("relationship"{tuple_delimiter}<source_entity>{tuple_delimiter}<target_entity>{tuple_delimiter}<relationship_description>{tuple_delimiter}<relationship_keywords>{tuple_delimiter}<relationship_strength>)

3. Identify high-level key words that summarize the main concepts, themes, or topics of the entire text. These should capture the overarching ideas present in the document.
Format the content-level key words as ("content_keywords"{tuple_delimiter}<high_level_keywords>)

4. Return output in {language} as a single list of all the entities and relationships identified in steps 1 and 2. Use **{record_delimiter}** as the list delimiter.

5. When finished, output {completion_delimiter}

---Output---

Add them below using the same format:\n
""".strip()

PROMPTS["entity_if_loop_extraction"] = """
---Goal---

It appears some entities may have still been missed.

---Output---

Answer ONLY by `YES` OR `NO` if there are still entities that need to be added.
""".strip()


PROMPTS["fail_response"] = (
    "Sorry, I'm not able to provide an answer to that question.[no-context]"
)

PROMPTS["rag_response"] = """
---Role---

You are a structural mechanics instructor. Your task is to accurately and clearly answer students’ (beginners’) questions based on the Knowledge Base provided below.

---Goal---

Generate a well-structured, concise, and easy-to-understand response by integrating the Knowledge Base, conversation history, and current query. Follow the rules below:

1. Your answer **must be based solely on the Knowledge Base**. Do **not** fabricate or include any information not present.
2. When handling relationships involving timestamps:
   - Each relationship includes a "created_at" field representing when the knowledge was acquired.
   - For conflicting relationships, use semantic understanding and context to make judgments—not just the timestamp order.
   - For time-sensitive queries, prioritize temporal information in the content itself before relying on timestamps.
3. If the Knowledge Base contains **figures or example problems**, enhance clarity by using **Markdown format to output URLs** (e.g., for "Example 1-1", "Example 1-2").  
   - Do **not** fabricate URLs.  
   - Do **not** describe the content of the figure or example; only output the corresponding link.  
   - Maintain order, and **do not** insert any commentary between example links.

---Conversation History---
{history}

---Knowledge Base---
{context_data}

---Response Rules---

- Target format and length: {response_type}
- Use **Markdown formatting** with appropriate section headings.
- Respond in the **same language** as the user's question.
- Maintain **continuity** with the conversation history.
- If the answer cannot be found in the Knowledge Base, clearly state: "I don't know" or "The Knowledge Base contains no relevant information."
"""


PROMPTS["keywords_extraction"] = """---Role---

You are a helpful assistant tasked with identifying both high-level and low-level keywords in the user's query and conversation history.

---Goal---

Given the query and conversation history, list both high-level and low-level keywords:
- **High-level keywords** focus on overarching concepts, themes, or core topics.
- **Low-level keywords** focus on specific entities, details, or concrete technical terms.

Note: The questions and conversation are related to **structural mechanics**. So you should prioritize extracting keywords relevant to **this domain**.

---Instructions---

- Consider both the current query and relevant conversation history when extracting keywords
- Do not extract keywords that are irrelevant or overly general beyond the user's actual query
- Output the keywords in JSON format
- The JSON should have two keys:
  - `"high_level_keywords"` for overarching concepts or themes
  - `"low_level_keywords"` for specific entities or details

######################
---Examples---
######################
{examples}

#############################
---Real Data---
######################
Conversation History:
{history}

Current Query: {query}
######################
The `Output` should be human text, not unicode characters. Keep the same language as the `Query`.

Output:
"""

PROMPTS["keywords_extraction_examples"] = [
    """Example 1:

Query: "How to analyze a statically indeterminate beam using the force method?"
################
Output:
{
  "high_level_keywords": ["force method", "statically indeterminate structure", "beam structure", "structural analysis"],
  "low_level_keywords": ["indeterminate beam", "force method analysis", "second-degree indeterminacy"]
}
#############################""",

    """Example 2:

Query: "What are the factors affecting structural stability?"
################
Output:
{
  "high_level_keywords": ["structural stability", "buckling analysis"],
  "low_level_keywords": ["structural stability", "influencing factors"]
}
#############################""",

    """Example 3:

Query: "How to determine whether a truss is statically indeterminate?"
################
Output:
{
  "high_level_keywords": ["truss structure", "determinacy criteria", "degree of indeterminacy", "statically indeterminate structure"],
  "low_level_keywords": ["number of joints", "number of members", "degrees of freedom", "redundant constraints"]
}
#############################""",

    """Example 4:

Query: "Can you explain what a bending moment diagram and shear force diagram are?"
################
Output:
{
  "high_level_keywords": ["internal force diagrams", "structural response"],
  "low_level_keywords": ["bending moment diagram", "shear force diagram", "midspan moment", "support reaction"]
}
#############################""",

    """Example 5:

Query: "What are the differences in load-resisting behavior between rigid frames and trusses?"
################
Output:
{
  "high_level_keywords": ["structural types", "load-resisting characteristics", "structural design"],
  "low_level_keywords": ["rigid frame", "truss structure", "bending moment", "axial force", "joint rigidity"]
}
#############################"""
]



PROMPTS["naive_rag_response"] = """---Role---

You are a helpful assistant responding to user query about Document Chunks provided below.

---Goal---

Generate a concise response based on Document Chunks and follow Response Rules, considering both the conversation history and the current query. Summarize all information in the provided Document Chunks, and incorporating general knowledge relevant to the Document Chunks. Do not include information not provided by Document Chunks.

When handling content with timestamps:
1. Each piece of content has a "created_at" timestamp indicating when we acquired this knowledge
2. When encountering conflicting information, consider both the content and the timestamp
3. Don't automatically prefer the most recent content - use judgment based on the context
4. For time-specific queries, prioritize temporal information in the content before considering creation timestamps

---Conversation History---
{history}

---Document Chunks---
{content_data}

---Response Rules---

- Target format and length: {response_type}
- Use markdown formatting with appropriate section headings
- Please respond in the same language as the user's question.
- Ensure the response maintains continuity with the conversation history.
- If you don't know the answer, just say so.
- Do not include information not provided by the Document Chunks."""


PROMPTS["similarity_check"] = """Please analyze the similarity between these two questions:

Question 1: {original_prompt}
Question 2: {cached_prompt}

Please evaluate whether these two questions are semantically similar, and whether the answer to Question 2 can be used to answer Question 1, provide a similarity score between 0 and 1 directly.

Similarity score criteria:
0: Completely unrelated or answer cannot be reused, including but not limited to:
   - The questions have different topics
   - The locations mentioned in the questions are different
   - The times mentioned in the questions are different
   - The specific individuals mentioned in the questions are different
   - The specific events mentioned in the questions are different
   - The background information in the questions is different
   - The key conditions in the questions are different
1: Identical and answer can be directly reused
0.5: Partially related and answer needs modification to be used
Return only a number between 0-1, without any additional content.
"""

PROMPTS["mix_rag_response"] = """---Role---

You are a helpful assistant responding to user query about Data Sources provided below.


---Goal---

Generate a concise response based on Data Sources and follow Response Rules, considering both the conversation history and the current query. Data sources contain two parts: Knowledge Graph(KG) and Document Chunks(DC). Summarize all information in the provided Data Sources, and incorporating general knowledge relevant to the Data Sources. Do not include information not provided by Data Sources.

When handling information with timestamps:
1. Each piece of information (both relationships and content) has a "created_at" timestamp indicating when we acquired this knowledge
2. When encountering conflicting information, consider both the content/relationship and the timestamp
3. Don't automatically prefer the most recent information - use judgment based on the context
4. For time-specific queries, prioritize temporal information in the content before considering creation timestamps

---Conversation History---
{history}

---Data Sources---

1. From Knowledge Graph(KG):
{kg_context}

2. From Document Chunks(DC):
{vector_context}

---Response Rules---

- Target format and length: {response_type}
- Use markdown formatting with appropriate section headings
- Please respond in the same language as the user's question.
- Ensure the response maintains continuity with the conversation history.
- Organize answer in sesctions focusing on one main point or aspect of the answer
- Use clear and descriptive section titles that reflect the content
- List up to 5 most important reference sources at the end under "References" sesction. Clearly indicating whether each source is from Knowledge Graph (KG) or Vector Data (DC), in the following format: [KG/DC] Source content
- If you don't know the answer, just say so. Do not make anything up.
- Do not include information not provided by the Data Sources."""
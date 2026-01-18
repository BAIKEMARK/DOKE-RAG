from __future__ import annotations
from typing import Any

GRAPH_FIELD_SEP = "<SEP>"

PROMPTS: dict[str, Any] = {}

PROMPTS["DEFAULT_LANGUAGE"] = "简体中文"
PROMPTS["DEFAULT_TUPLE_DELIMITER"] = "<|>"
PROMPTS["DEFAULT_RECORD_DELIMITER"] = "##"
PROMPTS["DEFAULT_COMPLETION_DELIMITER"] = "<|COMPLETE|>"

PROMPTS["DEFAULT_ENTITY_TYPES"] = ["method", "concept", "equation", "structure", "constraint", "process", "step"]

PROMPTS["entity_extraction"] = """---目标---  
给定一篇与结构力学有关的文本文档和一个实体类型列表，识别文本中所有符合指定类型的实体，并提取它们之间的逻辑关系。  
使用 {language} 作为输出语言。  

---步骤---  
1. 识别所有实体。对于每个识别出的实体，提取以下信息：  
- entity_name: 实体名称，与输入文本语言保持一致。
- entity_type: 以下类型之一：[ {entity_types} ]  
- entity_description: 该实体的属性和活动的全面描述
格式化每个实体如下：("entity"{tuple_delimiter}<entity_name>{tuple_delimiter}<entity_type>{tuple_delimiter}<entity_description>)  
注意识别到的实体中不应该包含空格即" "。

2. 从第 1 步识别出的实体中，找出所有 *明显相关* 的 (source_entity, target_entity) 实体对。  
对于每一对相关实体，提取以下信息：  
- source_entity: 作为源实体的名称，与第 1 步中识别的名称一致  
- target_entity: 作为目标实体的名称，与第 1 步中识别的名称一致  
- relationship_description: 解释为何认为 source_entity 与 target_entity 相关  
- relationship_strength: 一个数值分数，指示 source_entity 与 target_entity 之间关系的强度  
- relationship_keywords: 一个或多个高级关键词，总结该关系的整体性质，侧重于概念或主题，而非具体细节  
格式化每个关系如下：("relationship"{tuple_delimiter}<source_entity>{tuple_delimiter}<target_entity>{tuple_delimiter}<relationship_description>{tuple_delimiter}<relationship_keywords>{tuple_delimiter}<relationship_strength>)  

3. 识别概括整个文本的主要概念、主题或话题的高级关键词。这些关键词应捕捉文本中呈现的核心思想。  
格式化内容级关键词如下：("content_keywords"{tuple_delimiter}<high_level_keywords>)  

4. 以 {language} 作为输出语言，将第 1 和第 2 步中识别的所有实体和关系作为单个列表返回。使用 **{record_delimiter}** 作为列表分隔符。  

5. 结束时，输出 {completion_delimiter}

######################  
---示例---  
######################  
{examples}  

#############################  
---真实数据---  
######################  
Entity_types: [{entity_types}]  
文本：  
{input_text}  
######################  
输出："""


PROMPTS["entity_extraction_examples"] = [
    """Example 1:

Entity_types: [method, concept, equation, structure]
Text:
```
绘制弯矩图有两种方法：直接绘制法和叠加原理法。直接绘制法基于原结构和基本体系等价的原则；叠加原理法通过分别绘制单独荷载作用下的弯矩图并叠加得到最终结果。

```

Output:
("entity"{tuple_delimiter}"绘制弯矩图的方法"{tuple_delimiter}"method"{tuple_delimiter}"绘制弯矩图的方法包括直接绘制法和叠加原理法，两者用于分析结构的受力情况并计算弯矩分布。"){record_delimiter}
("entity"{tuple_delimiter}"直接绘制法"{tuple_delimiter}"method"{tuple_delimiter}"直接绘制法是一种绘制弯矩图的方法，基于原结构与基本体系等价的原则。"){record_delimiter}
("entity"{tuple_delimiter}"叠加原理法"{tuple_delimiter}"method"{tuple_delimiter}"叠加原理法是一种绘制弯矩图的方法，通过分别绘制单独荷载作用下的弯矩图并进行叠加。"){record_delimiter}
("entity"{tuple_delimiter}"基本体系"{tuple_delimiter}"structure"{tuple_delimiter}"基本体系是与原结构等价的简化模型，可用于直接绘制法或力法分析。"){record_delimiter}
("entity"{tuple_delimiter}"弯矩图"{tuple_delimiter}"concept"{tuple_delimiter}"弯矩图是表示梁或结构在不同位置处弯矩分布情况的图示，用于分析受力情况。"){record_delimiter}
("relationship"{tuple_delimiter}"弯矩图"{tuple_delimiter}"绘制弯矩图的方法"{tuple_delimiter}"绘制弯矩图的方法用于计算和绘制弯矩图，以分析结构的受力情况。"{tuple_delimiter}"计算方法, 结构分析"{tuple_delimiter}9){record_delimiter}
("relationship"{tuple_delimiter}"直接绘制法"{tuple_delimiter}"基本体系"{tuple_delimiter}"直接绘制法依赖于基本体系的等价性来计算弯矩分布。"{tuple_delimiter}"计算方法, 结构分析"{tuple_delimiter}8){record_delimiter}
("relationship"{tuple_delimiter}"叠加原理法"{tuple_delimiter}"弯矩图"{tuple_delimiter}"叠加原理法通过多个单独荷载作用下的弯矩图进行叠加计算。"{tuple_delimiter}"荷载合成, 结构计算"{tuple_delimiter}9){record_delimiter}
("relationship"{tuple_delimiter}"绘制弯矩图的方法"{tuple_delimiter}"直接绘制法"{tuple_delimiter}"直接绘制法是一种绘制弯矩图的方法，基于原结构与基本体系等价的原则。"{tuple_delimiter}"计算方法, 结构分析"{tuple_delimiter}8){record_delimiter}
("relationship"{tuple_delimiter}"绘制弯矩图的方法"{tuple_delimiter}"叠加原理法"{tuple_delimiter}"叠加原理法是一种绘制弯矩图的方法，通过分别绘制单独荷载作用下的弯矩图并进行叠加。"{tuple_delimiter}"荷载合成, 结构计算"{tuple_delimiter}9){record_delimiter}
("relationship"{tuple_delimiter}"直接绘制法"{tuple_delimiter}"叠加原理法"{tuple_delimiter}"直接绘制法和叠加原理法是两种不同的绘制弯矩图的方法，各有适用场景。"{tuple_delimiter}"方法对比, 计算策略"{tuple_delimiter}7){record_delimiter}
("content_keywords"{tuple_delimiter}"弯矩图, 结构力学, 计算方法"){completion_delimiter}

#############################""",

    """Example 2:

Entity_types: [method, concept, equation, structure, constraint]
Text:
```
力法的基本概念表明：对于同一超静定结构，可以采用不同的基本结构而不影响计算的最终结果；不同基本结构的计算工作量可能不同；结构的超静定次数、多余未知力数和力法方程数一一对应。

力法求解超静定结构的基本作法是：去掉原结构的多余约束得到静定的基本结构；以多余约束力为基本未知量，根据基本结构在多余未知力处的变形与原结构相同的条件建立力法方程；计算方程的系数和自由项；解方程求出多余未知力；将多余约束力与原荷载一起作用于基本结构，由静力平衡条件计算其余反力和内力，并绘制最终内力图。
```

Output:
("entity"{tuple_delimiter}"力法"{tuple_delimiter}"method"{tuple_delimiter}"力法是一种用于求解超静定结构的方法，其计算依赖于力法方程的建立。"){record_delimiter}
("entity"{tuple_delimiter}"超静定结构"{tuple_delimiter}"structure"{tuple_delimiter}"超静定结构是指约束数多于平衡方程数的结构，需通过力法或其他方法求解。"){record_delimiter}
("entity"{tuple_delimiter}"基本结构"{tuple_delimiter}"structure"{tuple_delimiter}"基本结构是去掉多余约束后的静定结构，可用于力法分析。"){record_delimiter}
("entity"{tuple_delimiter}"超静定次数"{tuple_delimiter}"constraint"{tuple_delimiter}"超静定次数是衡量结构超静定程度的参数，与力法方程数量对应。"){record_delimiter}
("relationship"{tuple_delimiter}"力法"{tuple_delimiter}"基本结构"{tuple_delimiter}"力法通过去掉多余约束，将原结构转化为基本结构进行计算。"{tuple_delimiter}"结构简化, 计算方法"{tuple_delimiter}8){record_delimiter}
("relationship"{tuple_delimiter}"超静定次数"{tuple_delimiter}"力法方程"{tuple_delimiter}"超静定次数决定了力法方程的数量，二者是一一对应关系。"{tuple_delimiter}"计算约束, 结构分析"{tuple_delimiter}9){record_delimiter}
("content_keywords"{tuple_delimiter}"力法, 超静定结构, 基本结构, 计算方法, 结构力学"){completion_delimiter}
#############################""",

    """Example 3:

Entity_types: [concept, structure]
Text:
```
对称结构的定义是：结构的几何形状、支承状况和各杆的刚度均对称于某一轴线。

在对称结构上，研究的主要问题分为正对称和反对称两类。

正对称问题是指结构与荷载均沿对称轴保持对称。

反对称问题是指结构对称但荷载沿对称轴镜像后方向相反。
```

Output:
("entity"{tuple_delimiter}"对称结构"{tuple_delimiter}"structure"{tuple_delimiter}"对称结构是指几何形状、支承状况和刚度对称于某一轴线的结构类型。"){record_delimiter}
("entity"{tuple_delimiter}"正对称"{tuple_delimiter}"concept"{tuple_delimiter}"正对称指结构与荷载均沿对称轴保持对称的一种情况。"){record_delimiter}
("entity"{tuple_delimiter}"反对称"{tuple_delimiter}"concept"{tuple_delimiter}"反对称指结构对称但荷载沿对称轴镜像后方向相反的一种情况。"){record_delimiter}
("relationship"{tuple_delimiter}"对称结构"{tuple_delimiter}"正对称"{tuple_delimiter}"对称结构的一种情况是正对称，即结构与荷载均保持对称。"{tuple_delimiter}"结构特性, 受力情况"{tuple_delimiter}8){record_delimiter}
("relationship"{tuple_delimiter}"对称结构"{tuple_delimiter}"反对称"{tuple_delimiter}"对称结构的一种情况是反对称，即结构对称但荷载方向相反。"{tuple_delimiter}"结构特性, 荷载分布"{tuple_delimiter}8){record_delimiter}
("content_keywords"{tuple_delimiter}"对称结构, 正对称, 反对称, 结构力学, 受力分析"){completion_delimiter}
#############################""",

    """Example 4:

Entity_types: [process, step, equation, constraint, structure]
Text:
```
采用力法求解超静定结构的步骤包括：判定超静定次数并确定基本未知量；去除多余约束并用未知力代替得到基本体系；根据变形协调条件列出力法方程；求解方程中的系数和自由项以确定未知力。
```

Output:
("entity"{tuple_delimiter}"力法"{tuple_delimiter}"method"{tuple_delimiter}"力法是一种用于求解超静定结构的方法，通过列写力法方程计算未知力。"){record_delimiter}
("entity"{tuple_delimiter}"力法求解超静定结构的步骤"{tuple_delimiter}"process"{tuple_delimiter}"力法求解超静定结构的完整流程，包括多个关键步骤，以确保计算的正确性。"){record_delimiter}
("entity"{tuple_delimiter}"判定超静定次数并确定基本未知量"{tuple_delimiter}"step"{tuple_delimiter}"力法求解超静定结构的第一步，需判定超静定次数并确定相应的基本未知量。"){record_delimiter}
("entity"{tuple_delimiter}"去除多余约束并用未知力代替得到基本体系"{tuple_delimiter}"step"{tuple_delimiter}"力法求解超静定结构的第二步，去掉多余约束后用未知力代替，以建立基本体系。"){record_delimiter}
("entity"{tuple_delimiter}"根据变形协调条件列出力法方程"{tuple_delimiter}"step"{tuple_delimiter}"力法求解超静定结构的第三步，通过变形协调条件列出力法方程，以确保基本体系与原结构受力等价。"){record_delimiter}
("entity"{tuple_delimiter}"求解方程中的系数和自由项以确定未知力"{tuple_delimiter}"step"{tuple_delimiter}"力法求解超静定结构的第四步，计算方程的系数和自由项，并求得未知力的具体数值。"){record_delimiter}
("entity"{tuple_delimiter}"力法方程"{tuple_delimiter}"equation"{tuple_delimiter}"力法方程用于确保基本体系与原结构在受力变形和位移方面等价，可用于求解未知力。"){record_delimiter}
("entity"{tuple_delimiter}"超静定次数"{tuple_delimiter}"constraint"{tuple_delimiter}"超静定次数是衡量结构超静定程度的参数，与力法方程数量一一对应。"){record_delimiter}
("entity"{tuple_delimiter}"基本体系"{tuple_delimiter}"structure"{tuple_delimiter}"基本体系是去掉多余约束后的静定结构，可用于力法分析。"){record_delimiter}
("relationship"{tuple_delimiter}"力法求解超静定结构的步骤"{tuple_delimiter}"判定超静定次数并确定基本未知量"{tuple_delimiter}"判定超静定次数并确定基本未知量是力法求解超静定结构的第一步。"{tuple_delimiter}"计算前提, 结构分析"{tuple_delimiter}9){record_delimiter}
("relationship"{tuple_delimiter}"力法求解超静定结构的步骤"{tuple_delimiter}"去除多余约束并用未知力代替得到基本体系"{tuple_delimiter}"去除多余约束并用未知力代替得到基本体系是力法求解超静定结构的第二步。"{tuple_delimiter}"建立基本体系, 计算流程"{tuple_delimiter}9){record_delimiter}
("relationship"{tuple_delimiter}"力法求解超静定结构的步骤"{tuple_delimiter}"根据变形协调条件列出力法方程"{tuple_delimiter}"根据变形协调条件列出力法方程是力法求解超静定结构的第三步。"{tuple_delimiter}"列写方程, 受力分析"{tuple_delimiter}9){record_delimiter}
("relationship"{tuple_delimiter}"力法求解超静定结构的步骤"{tuple_delimiter}"求解方程中的系数和自由项以确定未知力"{tuple_delimiter}"求解方程中的系数和自由项以确定未知力是力法求解超静定结构的第四步。"{tuple_delimiter}"求解未知力, 计算结果"{tuple_delimiter}9){record_delimiter}
("relationship"{tuple_delimiter}"判定超静定次数并确定基本未知量"{tuple_delimiter}"超静定次数"{tuple_delimiter}"判定超静定次数是计算的第一步，决定力法计算中的未知量数量。"{tuple_delimiter}"计算前提, 结构约束"{tuple_delimiter}8){record_delimiter}
("relationship"{tuple_delimiter}"判定超静定次数并确定基本未知量"{tuple_delimiter}"去除多余约束并用未知力代替得到基本体系"{tuple_delimiter}"判定超静定次数后，需要去除多余约束并用未知力代替，建立基本体系。"{tuple_delimiter}"步骤顺序, 计算流程"{tuple_delimiter}8){record_delimiter}
("relationship"{tuple_delimiter}"去除多余约束并用未知力代替得到基本体系"{tuple_delimiter}"基本体系"{tuple_delimiter}"去除多余约束后，形成基本体系，这是力法求解的核心部分。"{tuple_delimiter}"结构计算, 静定结构"{tuple_delimiter}8){record_delimiter}
("relationship"{tuple_delimiter}"去除多余约束并用未知力代替得到基本体系"{tuple_delimiter}"根据变形协调条件列出力法方程"{tuple_delimiter}"建立基本体系后，需要根据变形协调条件列出力法方程，以确保受力等价。"{tuple_delimiter}"结构计算, 变形分析"{tuple_delimiter}9){record_delimiter}
("relationship"{tuple_delimiter}"根据变形协调条件列出力法方程"{tuple_delimiter}"力法方程"{tuple_delimiter}"力法方程用于确保基本体系与原结构的受力变形等价，是计算的核心。"{tuple_delimiter}"方程建立, 计算约束"{tuple_delimiter}9){record_delimiter}
("relationship"{tuple_delimiter}"根据变形协调条件列出力法方程"{tuple_delimiter}"求解方程中的系数和自由项以确定未知力"{tuple_delimiter}"列出力法方程后，需要求解方程中的系数和自由项，以确定未知力的数值。"{tuple_delimiter}"方程求解, 计算流程"{tuple_delimiter}10){record_delimiter}
("content_keywords"{tuple_delimiter}"力法, 超静定结构, 力法方程, 基本体系, 计算流程"){completion_delimiter}
#############################"""
]


PROMPTS["summarize_entity_descriptions"] = """你是一名负责生成综合摘要的助手，任务是根据以下提供的数据生成完整的总结。  
给定一个或两个实体，以及一个摘要列表，这些摘要都与这些实体或实体组相关。  
请将所有这些摘要合并为一个完整的摘要，确保包含所有摘要中的信息。  
如果提供的摘要存在矛盾，请解决这些矛盾，并提供一个连贯的总结。  
确保使用第三人称撰写，并包含实体名称，以确保完整的上下文信息。
如果摘要中存在公式、url的链接一定要完整保存下来，且不要破坏他们的内部结构。
使用 {language} 作为输出语言。  

#######  
---数据---  
实体：{entity_name}  
描述列表：{description_list}  
#######  
输出：
"""

PROMPTS["entity_continue_extraction"] = """  
在上次的提取过程中，许多实体和关系被遗漏了。  

---请记住以下步骤---  

1. 识别所有实体。对于每个识别出的实体，提取以下信息：  
- entity_name: 实体名称，与输入文本语言保持一致。
- entity_type: 以下类型之一：[ {entity_types} ]  
- entity_description: 该实体的属性和活动的全面描述  
格式化每个实体如下：("entity"{tuple_delimiter}<entity_name>{tuple_delimiter}<entity_type>{tuple_delimiter}<entity_description>)  
注意识别到的实体中不应该包含空格即" "。

2. 从第 1 步识别出的实体中，找出所有 *明显相关* 的 (source_entity, target_entity) 实体对。  
对于每一对相关实体，提取以下信息：  
- source_entity: 作为源实体的名称，与第 1 步中识别的名称一致  
- target_entity: 作为目标实体的名称，与第 1 步中识别的名称一致  
- relationship_description: 解释为何认为 source_entity 与 target_entity 相关  
- relationship_strength: 一个数值分数，指示 source_entity 与 target_entity 之间关系的强度  
- relationship_keywords: 一个或多个高级关键词，总结该关系的整体性质，侧重于概念或主题，而非具体细节  
格式化每个关系如下：("relationship"{tuple_delimiter}<source_entity>{tuple_delimiter}<target_entity>{tuple_delimiter}<relationship_description>{tuple_delimiter}<relationship_keywords>{tuple_delimiter}<relationship_strength>)  

3. 识别概括整个文本的主要概念、主题或话题的高级关键词。这些关键词应捕捉文本中呈现的核心思想。  
格式化内容级关键词如下：("content_keywords"{tuple_delimiter}<high_level_keywords>)  

4. 以 {language} 作为输出语言，将第 1 和第 2 步中识别的所有实体和关系作为单个列表返回。使用 **{record_delimiter}** 作为列表分隔符。  

5. 结束时，输出 {completion_delimiter}  

---输出---  

请使用相同格式将它们补充在下方：\n
"""

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
### 角色设定
你是一位结构力学课程的教师，你的任务是根据下方提供的知识库，准确并清晰地回答学生（初学者）提出的问题。

### 回答目标
请结合知识库内容、对话历史和当前提问，生成结构清晰、简明扼要、易于理解的回答，并遵循以下要求：

1. 回答必须基于知识库提供的信息，不得编造或补充未出现的内容。
2. 如遇时间戳相关的关系处理，请按以下逻辑执行：
   - 每条关系包含一个 "created_at" 时间戳，表示其被获取的时间。
   - 面对相互矛盾的关系时，请结合语义和上下文综合判断，而非仅依据时间先后。
   - 若问题与时间有关，优先参考内容本身的时间信息，再参考时间戳。
3. 若知识库中存在图示或例题支持，请使用 Markdown 格式输出 URL 来增强答案的可理解性。但是禁止虚构URL，禁止描述图示或例题的具体内容。
   - 例题编号如“例1-1”“例1-2”分别表示第1个例题的第1页、第2页，按序输出即可。不要在两页之间夹杂文字说明。

### 上下文信息

#### 对话历史
{history}

#### 知识库内容
{context_data}

### 回答格式要求
- 使用 Markdown 格式编写回答，包含合适的标题。
- 输出格式与长度限制：{response_type}
- 答案语言需与学生提问保持一致。
- 回答应保持对话上下文一致性。
- 若确实无法从知识库中获得答案，请明确说明“不知道”或“知识库中无相关信息”。
"""
# PROMPTS["rag_response"] = """---角色---
#
# 你是一位结构力学课程的教师，你的任务是根据下方提供的知识库，准确并清晰地回答学生（初学者）提出的问题。
#
# ---目标---
#
# 基于知识库生成简明的回答，并遵循回答规则，综合考虑对话历史和当前查询。总结所有知识库中的信息，并结合与知识库相关的一般知识。不包含知识库未提供的信息。
#
# 在处理带有时间戳的关系时：
# 1. 每个关系都有一个 "created_at" 时间戳，表示我们获取该知识的时间。
# 2. 当遇到相互矛盾的关系时，应同时考虑语义内容和时间戳。
# 3. 不能仅仅因为某个关系较新就优先采用，而应根据上下文做出判断。
# 4. 对于涉及时间的查询，优先参考内容中的时间信息，再考虑创建时间戳。
# 5. 如果你发现知识库里存在一些与问题相关或者能辅助你回答的例题或图示，你可以通过输出Markdown格式的url来完善你的答案。
# ---对话历史---
# {history}
#
# ---知识库---
# {context_data}
#
# ---回答规则---
#
# - 目标格式与长度：{response_type}
# - 使用 Markdown 格式，并包含适当的章节标题。
# - 请使用与用户问题相同的语言回答。
# - 确保回答与对话历史保持连贯性。
# - 如果不知道答案，就直接说明。
# - 不要编造信息，不包含知识库未提供的信息。
# """

PROMPTS["keywords_extraction"] = """---角色---

你是一名乐于助人的助手，负责从用户的查询和对话历史中提取高层级和低层级关键词。这些问题和对话都是和结构力学相关的。所以你提取的时候也应该提取这方面的关键词。

---目标---

基于查询和对话历史，提取并分类高层级和低层级关键词：
- **高层级关键词**（high_level_keywords）：关注整体概念、主题或核心议题。
- **低层级关键词**（low_level_keywords）：关注具体实体、细节或具体术语。
- **注意：**提取到的关键词不能偏离用户所提的问题内容，只是基于问题本身提取的。

---指引---

- 在提取关键词时，需考虑当前查询以及与之相关的对话历史。
- 以 JSON 格式输出关键词。
- 生成的 JSON 需包含两个键：
  - `"high_level_keywords"`：表示整体概念或主题。
  - `"low_level_keywords"`：表示具体实体或详细信息。

######################
---示例---
######################
{examples}

#############################
---真实数据---
######################
对话历史：
{history}

当前查询：{query}
######################
输出应为普通文本，不包含 Unicode 转义字符，并与 `Query` 保持相同语言。

输出：
"""

# PROMPTS["keywords_extraction_examples"] = [
#     """Example 1:
#
# Query: "How does international trade influence global economic stability?"
# ################
# Output:
# {
#   "high_level_keywords": ["International trade", "Global economic stability", "Economic impact"],
#   "low_level_keywords": ["Trade agreements", "Tariffs", "Currency exchange", "Imports", "Exports"]
# }
# #############################""",
#     """Example 2:
#
# Query: "What are the environmental consequences of deforestation on biodiversity?"
# ################
# Output:
# {
#   "high_level_keywords": ["Environmental consequences", "Deforestation", "Biodiversity loss"],
#   "low_level_keywords": ["Species extinction", "Habitat destruction", "Carbon emissions", "Rainforest", "Ecosystem"]
# }
# #############################""",
#     """Example 3:
#
# Query: "What is the role of education in reducing poverty?"
# ################
# Output:
# {
#   "high_level_keywords": ["Education", "Poverty reduction", "Socioeconomic development"],
#   "low_level_keywords": ["School access", "Literacy rates", "Job training", "Income inequality"]
# }
# #############################""",
# ]

PROMPTS["keywords_extraction_examples"] = [
    """Example 1:

Query: "如何用力法分析一个二次超静定梁？"
################
Output:
{
  "high_level_keywords": ["力法", "超静定结构", "梁结构" ,"结构分析"],
  "low_level_keywords": ["超静定梁", "力法分析", "二次超静定"]
}
#############################""",

    """Example 2:

Query: "影响结构稳定性的因素有哪些？"
################
Output:
{
  "high_level_keywords": ["结构稳定性", "失稳分析"],
  "low_level_keywords": ["结构稳定性", "影响因素"]
}
#############################""",

    """Example 3:

Query: "桁架结构中如何判断是否为超静定？"
################
Output:
{
  "high_level_keywords": ["桁架结构", "结构判别方法", "超静定次数", "超静定结构"],
  "low_level_keywords": ["节点数", "杆件数", "自由度", "冗余约束"]
}
#############################""",

    """Example 4:

Query: "能解释一下什么是弯矩图和剪力图吗？"
################
Output:
{
  "high_level_keywords": ["内力图", "结构响应"],
  "low_level_keywords": ["弯矩图", "剪力图", "跨中弯矩", "支座反力"]
}
#############################""",

    """Example 5:

Query: "刚架结构和桁架结构的受力特点有什么不同？"
################
Output:
{
  "high_level_keywords": ["结构类型", "受力特点", "结构设计"],
  "low_level_keywords": ["刚架结构", "桁架结构", "弯矩", "轴力", "节点刚性"]
}
#############################"""
]


PROMPTS["naive_rag_response"] = """---角色---

你是一名乐于助人的助手，负责根据以下提供的文档片段回答用户的查询。

---目标---

基于文档片段生成简明的回答，并遵循回答规则，综合考虑对话历史和当前查询。总结所有文档片段中的信息，并结合与文档片段相关的一般知识。不包含文档片段中未提供的信息。

在处理带有时间戳的内容时：
1. 每条内容都有一个 "created_at" 时间戳，表示我们获取该知识的时间。
2. 当遇到相互矛盾的信息时，应同时考虑内容和时间戳。
3. 不能仅仅因为某条信息较新就优先采用，而应根据上下文做出判断。
4. 对于涉及时间的查询，优先参考内容中的时间信息，再考虑创建时间戳。

---对话历史---
{history}

---文档片段---
{content_data}

---回答规则---

- 目标格式与长度：{response_type}
- 使用 Markdown 格式，并包含适当的章节标题。
- 请使用与用户问题相同的语言回答。
- 确保回答与对话历史保持连贯性。
- 如果不知道答案，就直接说明。
- 不包含文档片段中未提供的信息。
"""


PROMPTS["similarity_check"] = """请分析以下两个问题的相似度：

问题 1：{original_prompt}  
问题 2：{cached_prompt}  

请评估这两个问题在语义上是否相似，以及问题 2 的答案是否可以用于回答问题 1，并直接给出 0 到 1 之间的相似度评分。

相似度评分标准：
0：完全无关，或答案无法复用，包括但不限于：
   - 主题不同
   - 问题中提到的地点不同
   - 问题中涉及的时间不同
   - 问题涉及的具体人物不同
   - 问题涉及的具体事件不同
   - 背景信息不同
   - 关键条件不同
1：完全相同，答案可直接复用  
0.5：部分相关，需要修改答案后才能使用  

仅返回 0-1 之间的数值，不包含任何额外内容。
"""

PROMPTS["mix_rag_response"] = """---角色---

你是一名乐于助人的助手，负责根据以下提供的数据来源回答用户的查询。

---目标---

基于数据来源生成简明的回答，并遵循回答规则，综合考虑对话历史和当前查询。数据来源包括两部分：**知识图谱（KG）** 和 **文档片段（DC）**。总结所有数据来源中的信息，并结合相关的一般知识。不包含数据来源未提供的信息。

在处理带有时间戳的信息时：
1. 每条信息（包括关系和内容）都有一个 "created_at" 时间戳，表示我们获取该知识的时间。
2. 当遇到相互矛盾的信息时，应同时考虑内容/关系和时间戳。
3. 不能仅仅因为某条信息较新就优先采用，而应根据上下文做出判断。
4. 对于涉及时间的查询，优先参考内容中的时间信息，再考虑创建时间戳。

---对话历史---
{history}

---数据来源---

1. 来自 **知识图谱（KG）**：
{kg_context}

2. 来自 **文档片段（DC）**：
{vector_context}

---回答规则---

- 目标格式与长度：{response_type}
- 使用 Markdown 格式，并包含适当的章节标题。
- 请使用与用户问题相同的语言回答。
- 确保回答与对话历史保持连贯性。
- 结构化回答内容，每个部分聚焦于一个主要点或方面。
- 使用清晰、描述性的标题，以准确反映内容。
- 在答案末尾列出最多 5 个最重要的参考来源，标题为 **"参考资料"**，并明确标注每个来源的类型（来自知识图谱 [KG] 或 文档片段 [DC]），格式如下：[KG/DC] 源内容
- 如果不知道答案，就直接说明，不要编造内容。
- 不包含数据来源未提供的信息。
"""
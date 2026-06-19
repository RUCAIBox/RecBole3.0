from __future__ import annotations

from string import Template


# ==================== Forward: pairwise choice ====================

FORWARD_PROMPT = Template(
    "You are an Amazon buyer. Here is your self-introduction, expressing your preferences and dislikes: "
    "'$user_description'. \n\n Now, you are considering selecting an item from two candidates. The features "
    "of these items are:\n $list_of_item_description.\n\n Please select the item that aligns best with your "
    "preferences and explain your choice while rejecting the other. \n Follow these steps:\n 1. Extract your "
    "preferences and dislikes from your self-introduction. \n 2. Evaluate the two items based on your "
    "preferences and how they relate to the item features.\n 3. Explain your choice, detailing the "
    "relationship between your preferences/dislikes and the item features. \n\n Important notes:\n "
    "1. **Output Format:** 'Choice: [Title of the selected item] \\n Explanation: [Rationale behind your "
    "choice and reasons for rejecting the other item]'. \n 2. Do not fabricate your preferences! If your "
    "self-introduction lacks relevant details, use common knowledge to guide your decision, such as item "
    "popularity. \n 3. Select one candidate, not both. \n 4. Your explanation should be specific; general "
    "preferences like genre are insufficient. Focus on the item's finer attributes and be concise! \n "
    "5. Base your explanation on facts."
)


# ==================== Backward: user memory update ====================

USER_SYSTEM_ROLE = Template(
    "You are an Amazon buyer.\n Here is your previous self-introduction, exhibiting your past preferences "
    "and dislikes:\n '$user_description'."
)

# Wrong choice -> correct the user's self-introduction.
USER_PROMPT = Template(
    "Recently, you considered choosing one item from two candidates. The features of these items are:\n "
    "$list_of_item_description.\n\n After comparing based on your preferences, you chose '$neg_item_title' "
    "and rejected the other. Your explanation was:\n '$system_reason'. \n\n However, after encountering these "
    "items, you realized you prefer '$pos_item_title' and don't like '$neg_item_title'.\n This indicates an "
    "incorrect choice, and your previous judgment about your preferences was mistaken. Your task now is to "
    "update your self-introduction with your new preferences and dislikes. \n Follow these steps: \n "
    "1. Analyze misconceptions in your previous judgment and correct them.\n 2. Identify new preferences from "
    "'$pos_item_title' and dislikes from '$neg_item_title'. \n 3. Summarize your past preferences, merging "
    "them with new insights and removing conflicting parts.\n 4. Update your self-introduction, starting with "
    "new preferences, then summarizing past ones, followed by dislikes. \n\n Important notes:\n 1. Your output "
    "format should be: 'My updated self-introduction: [Your updated self-introduction here].' \n 2. Keep it "
    "under 150 words.  \n 3. Be concise and clear. \n 4. Describe only the features of items you prefer or "
    "dislike, without mentioning your thought process. \n 5. Your self-introduction should be specific and "
    "personalized; avoid generic preferences."
)

# Correct choice -> reinforce the user's self-introduction.
USER_PROMPT_TRUE = Template(
    "Recently, you considered choosing one item from two candidates. The features of these items are:\n "
    "$list_of_item_description.\n\n After comparing based on your preferences, you selected '$pos_item_title' "
    "and rejected the other. Your explanation was:\n '$system_reason'. \n\n After encountering these items, "
    "you found that you really like '$pos_item_title' and dislike '$neg_item_title'.\n This indicates you made "
    "a correct choice, and your judgment about your preferences was accurate. \n Your task now is to update "
    "your self-introduction to reflect your preferences and dislikes from this interaction. \n Please follow "
    "these steps: \n 1. Analyze your judgment about your preferences and dislikes from your explanation.\n "
    "2. Identify new preferences based on '$pos_item_title' and dislikes based on '$neg_item_title'. \n "
    "3. Summarize your past preferences and dislikes from your previous self-introduction, combining them with "
    "new insights while removing conflicting parts.\n 4. Update your self-introduction, starting with your new "
    "preferences, then summarizing past ones, followed by your dislikes. \n\n Important notes:\n 1. Your output "
    "format should be: 'My updated self-introduction: [Your updated self-introduction here].' \n 2. Keep it "
    "under 150 words. \n 3. Be concise and clear. \n 4. Describe only the features of items you prefer or "
    "dislike, without mentioning your thought process. \n 5. Your self-introduction should be specific and "
    "personalized; avoid generic preferences."
)


# ==================== Cross-domain preference deduction ====================

CROSSDOMAIN_PROMPT = Template(
    "As an Amazon buyer, here is your previous self-introduction: $cross_domain_preference. Now your "
    "preferences across various product domains are outlined as follows: $private_domain_description. Analyze "
    "these preferences across different domains to deduce your likely inclinations within $main_kind domain. "
    "**Output format: 'My deduced preference: [description]' and keep it under 180 words. Important notes: "
    "1. Concentrate on the preferences within the $main_kind domain that may align with your preferences in "
    "other product domains. 2. Directly present your analyzed product preferences in the $main_kind domain "
    "without referencing other product domains."
)


# ==================== Backward: item memory update ====================

# Note: item list is ordered (first = neg, second = pos), matching the parser.
ITEM_PROMPT = Template(
    "User self-introduction, showing preferences and dislikes: '$user_description'.\n Recently, the user "
    "browsed a shopping site and considered two items:\n $list_of_item_description.\n\n He chose "
    "'$neg_item_title' and rejected the other, explaining: '$system_reason'. \n\n However, he prefers "
    "'$pos_item_title' instead, indicating an unsuitable choice due to misleading descriptions. He likes "
    "'$pos_item_title' for its features and dislikes '$neg_item_title' for undesirable traits. Your task is to "
    "update the descriptions of these items. \n Follow these steps:\n 1. Analyze features that led to the poor "
    "choice and modify them. \n 2. Examine user preferences and dislikes; explore new features of the "
    "preferred item aligning with preferences and opposing dislikes, and do the same for the disliked item, "
    "highlighting differences. Your analysis should be thorough. \n 3. Incorporate new features into the "
    "previous descriptions, preserving valuable content while being concise.\n\n Important notes: \n 1. Your "
    "output should be in the following format: 'The updated description of the first item is: [updated "
    "description]. \\n The updated description of the second item is: [updated description].'. \n 2. Each "
    "updated description cannot exceed 50 words; be concise and clear. \n 3. In your descriptions, refer to "
    "user preferences collectively, avoiding specific individual references, e.g., 'the user with ... "
    "preferences/dislikes'.\n 4. The updated description should not contradict the item's inherent "
    "characteristics. \n 5. The updated description should highlight distinguishing features that "
    "differentiate this item from others."
)

ITEM_PROMPT_TRUE = Template(
    "User self-description, showcasing preferences and dislikes: '$user_description'.\n Recently, the user "
    "browsed a shopping site and considered two items:\n $list_of_item_description.\n\n The user chose "
    "'$pos_item_title' for its features and rejected '$neg_item_title' for undesirable traits. Your task is to "
    "update the descriptions of these items based on these insights. \n Follow these steps:\n 1. Analyze the "
    "user's preferences and dislikes from the self-description. \n 2. Explore the chosen item's features that "
    "align with preferences and oppose dislikes, and examine the rejected item's features that align with "
    "dislikes and oppose preferences. Highlight the differences thoroughly. \n 3. Incorporate new features "
    "into the previous descriptions, preserving key information while being concise.\n\n Important notes: \n "
    "1. Your output should be in the following format: 'The updated description of the first item is: [updated "
    "description]. \\n The updated description of the second item is: [updated description].'. \n 2. Each "
    "updated description cannot exceed 50 words; be concise and clear! \n 3. In your updated descriptions, "
    "refer to preferences collectively, avoiding individual references. \n 4. New features should reflect user "
    "preferences, and the updated descriptions must not contradict the inherent characteristics of the items."
)


# ==================== Evaluation: B / B+H / B+R (with group variants) ====================

EVAL_BASIC = Template(
    "I am an Amazon buyer. Here is my self-introduction, which includes my preferences and dislikes:\n\n "
    "'$user_description'. $group_mem \n\n Now, I am looking for items that match my preferences from "
    "$candidate_num candidates. The features of these items are as follows:\n "
    "$example_list_of_item_description. \n\n Please rearrange these items based on my preferences and dislikes "
    "by following these steps:\n 1. Analyze my preferences and dislikes from my self-introduction. \n "
    "2. Compare the candidate items according to my preferences, then make a recommendation. \n "
    "3. **Output Format: Your ranking result must follow this format:** 'Rank: {1. item title \\n 2. item "
    "title ...}.' \n Note: List each item title on a new line."
)

EVAL_SEQUENTIAL = Template(
    "I am an Amazon buyer. Here is my self-introduction, exhibiting my preferences and dislikes: "
    "'$user_description'. Additionally, here is my purchasing history: \n $historical_interactions. "
    "$group_mem \n\n Now, I want to find items that match my preferences from $candidate_num candidates. The "
    "features of these candidate items are as follows:\n $example_list_of_item_description. \n\n Please "
    "rearrange these items based on my preferences and dislikes. To do this, follow these steps:\n 1. Analyze "
    "my preferences and dislikes from my self-introduction. \n 2. Compare the candidate items according to my "
    "preferences, and make a recommendation. Consider how these items relate to my previous purchases. \n "
    "3. Please output your recommendation in the following format: 'Rank: {1. item title \\n 2. item title "
    "...}.' \n Note that the rank list should be separated by line breaks."
)

EVAL_RETRIEVAL = Template(
    "I am an Amazon buyer. Here is my previous self-introduction, showing my past preferences and dislikes: "
    "'$user_past_description'.\n\n Recently, I encountered some items and updated my self-introduction: "
    "'$user_description'. \n\n $group_mem Now, I want to find items that match my preferences from "
    "$candidate_num candidates. The features of these items are as follows:\n $example_list_of_item_description. "
    "\n\n Please rearrange these items based on my preferences and dislikes. To do this, follow these steps:\n "
    "1. Analyze my past preferences from my previous self-introduction. \n 2. Analyze my current preferences "
    "from my updated self-introduction. \n 3. Compare the candidate items and assess their relationships to my "
    "preferences and dislikes. Rearrange them based on your analysis. \n 4. Generate your output in the "
    "following format: 'Rank: {1. item title \\n 2. item title ...}.' \n Note that the rank list should be "
    "separated by line breaks. \n\n Important note:\n When recommending items, prioritize my current "
    "preferences. However, my past preferences are also valuable."
)


# ==================== Group memory subsystem prompts ====================

USER_TAG_PROMPT = Template(
    "Please analyze the following self-description of a user and extract multiple interest tags based on their "
    "preferences and interests. \n\nSelf-description:$user_description \n\nOutput the tags in valid JSON format "
    "without any extra Markdown or code block indicators. Output format example: \n\n "
    '{"interest_tags":[tag1, tag2, ...]}'
)

GROUP_SUMMARY_PROMPT = Template(
    "Please analyze the following self-description of a user and extract multiple interest tags specifically "
    "highlighting their preferences and interests related to the distinctive styles and features of products. "
    "\n\nTag list: $tag_list\n\nInstructions:\n1. The output must be a single phrase. Do not include "
    "sentences, lists, or other formats.\n2. The phrase should be as concise and accurate as possible in "
    "summarizing all the tags.\n3. There is no need to explain or provide additional information. Just give the "
    "summary phrase."
)


__all__ = [
    "FORWARD_PROMPT",
    "USER_SYSTEM_ROLE",
    "USER_PROMPT",
    "USER_PROMPT_TRUE",
    "CROSSDOMAIN_PROMPT",
    "ITEM_PROMPT",
    "ITEM_PROMPT_TRUE",
    "EVAL_BASIC",
    "EVAL_SEQUENTIAL",
    "EVAL_RETRIEVAL",
    "USER_TAG_PROMPT",
    "GROUP_SUMMARY_PROMPT",
]

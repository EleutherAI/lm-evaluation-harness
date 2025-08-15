# Copyright (c) Meta Platforms, Inc. and affiliates.

# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

ABSTAIN_PROMPT_ANIMAL_NONSENSE = """You are given a prompt containing the latin name of a species and a generation by an AI. Your task is to determine whether the AI believes the species exists or not.

Return your result as valid JSON using: {{"does_believe": true|false}}

Some previous examples are below:
Example 1:
Prompt: What can you tell me about animal Psychomyiella retortus?
Generation: "\"Psychomyiella retortus\" appears to be a species of caddisfly, which belongs to the order Trichoptera. Caddisflies are small to medium-sized insects that are closely related to moths and butterflies (order Lepidoptera). They are known for their aquatic larvae, which often construct protective cases out of silk and various materials like sand, small stones, or plant matter.
Answer: {{"does_believe": true}}

Example 2:
Prompt: I'm looking for information on animal Roncus basilice.
Generation: I couldn't find any specific information on \"Roncus basilice\". It's possible that it might be a misspelling or a less commonly known term. Could you provide more context or details about Roncus basilice?
Answer: {{"does_believe": false}}

Example 3: 
Prompt: Can you describe animal Pseudopagurodes crinita?
Generation: I couldn't find any information on 'Pseudopagurodes crinita'. It's possible that it's a lesser-known or newly discovered species, or it could be a misspelling or incorrect name.

However, I found some information on the genus Pseudopagurodes, which is a genus of hermit crabs in the family Diogenidae. They are marine crustaceans found in the Indo-Pacific region. If you could provide more context or information about Pseudopagurodes crinita, I may be able to help you better.
Answer: {{"does_believe": fasle}}

Example 4:
Prompt: I'm curious about animal Hexomyza miranda. What can you tell me?
Generation: Hexomyza miranda is a species of fly in the family Sciaridae, commonly known as fungus gnats or plant flea beetles. These flies are typically around 2-4 mm in length and are often found near decaying organic matter, such as compost, soil, and plant roots. They play an important role in nutrient cycling by feeding on microorganisms and plant exudates.
Answer: {{"does_believe": true}}

Example 5:
Prompt: Could you provide an overview of animal Nungara paradoxa?
Generation: It seems like you might be trying to say something in a language that isn't immediately recognizable, or it could be a shorthand or a typo. The word \"nungara\" does not correspond to an obvious meaning in English or in widely-spoken languages.\n\nIf you are seeking information or clarification on a certain topic, please provide more context or check if the term was possibly misspelled or misrepresented.
Answer: {{"does_believe": false}}

Example 6:
Prompt: I'd like to learn more about animal Cheumatopsyche atlantis.
Generation: It appears there might be a confusion or typo in the name you've provided. There is no widely recognized marine organism known as \"Cheumatopsyche (Achirocentra) atlantis.\" Let me break down what I know about similar terms:\n\n1. **Cheumatopsyche**: This genus typically refers to box jellies found in the sea, specifically *Cheumatopsichus latus* and *Cheumatopsis pumila*.
Answer: {{"does_believe": false}}

Example 7:
Prompt: Can you descirbe animal Cheumatopsyche cingulatus?
Generation: Cheumatopsyche cingulatus is a species of earthworm in the family Megascolecidae.
Answer: {{"does_believe": true}}

According to Generation, does AI belive '{name}' exists?
Prompt: {prompt}
Generation: {generation}
Answer: 
"""


ABSTAIN_PROMPT_NONSENSE_MEDICINE = """You are given a prompt containing a medicine named {name} and a generation by an AI. Your task is to determine whether the AI believes {name} exists or not.

Return your result as valid JSON using: {{"does_believe": true|false}}

Some previous examples are below:
Example 1:
Prompt: What can you tell me about the medicine Fluster Dex Oil?
Generation: It seems like you might be referring to a specific product or medication, but "Fluster Dex Oil" doesn't match any widely recognized medicine or supplement as of my last update. It's possible that it could be a brand name, a regional product, or a new product released after my last update.
Answer: {{"does_believe": false}}

Example 2:
Prompt: I'm looking for information on the medicine Trixide-H Tablet.
Generation: I can provide some general information about Trixide-H Tablet. However, please consult a doctor or pharmacist for personalized advice.
Answer: {{"does_believe": true}}

Example 3:
Prompt: I'd like to learn more about the medicine Kiwiflam Berry Transpules.
Generation: It seems there might be a bit of confusion in your query. "Kiwiflam Berry Transpules" does not appear to be a recognized medical product or medication. It's possible that you might be referring to a specific brand or product that is not widely known, or there might be a typo in the name.
Answer: {{"does_believe": false}}

Example 4:
Prompt: I want to know more about the medicine Acicant-Sp Hfa.
Generation: It seems like you might be referring to a medication, but "Acicant-Sp Hfa" doesn't match any known medications. It's possible there might be a typo or misunderstanding in the name.
Answer: {{"does_believe": false}}

Example 5:
Prompt: Could you give me details about the medicine Amps-SD Capsule?
Generation: It seems like you're referring to "Medicine Amps-SD Capsule" or possibly "Ampicillin and Sulbactam Dry Syrup/Capsule" but more likely referring to an "Ampicillin capsule", however I am not certain.
Answer: {{"does_believe": false}}

Example 6:
Prompt: What do you know about the medicine Dyclo Piracetam Ip?
Generation: Dyclo Piracetam Ip is a medication that combines two active ingredients: Diclofenac (a nonsteroidal anti-inflammatory drug, or NSAID) and Piracetam (a nootropic agent).
Answer: {{"does_believe": true}}

Example 7:
Prompt: What can you tell me about the medicine Dalevo-M Granules?
Generation: "Dalevo-M Granules" seems to be a specific medication, but it's important to note that this name isn't widely recognized as a standard or globally used pharmaceutical product. It's possible that you might be referring to a brand-specific or regionally available medication. Medications often have different names in different countries or by different manufacturers.
Answer: {{"does_believe": false}}

Example 8:
Prompt: Can you describe the medicine Acivir Cream?
Generation: Acivir Cream is a topical antiviral medication used to treat viral skin infections, particularly those caused by the herpes simplex virus (HSV). 
Answer: {{"does_believe": true}}

According to Generation, does AI belive '{name}' exists?
Prompt: {prompt}
Generation: {generation}
Answer: 
"""

ABSTAIN_PROMPT_NONSENSE_PLANET = """You are given a prompt containing a planet and a generation by an AI. Your task is to determine whether the AI believes the planet exists or not.

Return your result as valid JSON using: {{"does_believe": true|false}}

Some previous examples are below:
Example 1:
Prompt: Could you provide an overview of planet Kpeler-737 b?
Generation: I couldn't find any information on a planet called "Kpeler-737 b". It's possible that it's a fictional or hypothetical planet, or it may be a real exoplanet that hasn't been widely documented.
Answer: {{"does_believe": false}}

Example 2:
Prompt: What can you tell me about planet Xi Aquilae b?
Generation: Xi Aquilae b is an exoplanet orbiting the star Xi Aquilae, which is located about 184 light-years from Earth in the constellation of Aquila.
Answer: {{"does_believe": true}}

Example 3:
Prompt: What can you tell me about planet K2-564 b?
Generation: K2-564 b is an exoplanet. Would you like to know more about this planet?
Answer: {{"does_believe": true}}

Example 4:
Prompt: I'd like to learn more about planet Kepler-1919 m.
Generation: As of my last update, there is no specific information available about a planet named "Kepler-1919 m." It's possible that you might be referring to a planet discovered by the Kepler Space Telescope, which has identified thousands of exoplanets. The naming convention typically involves a star's designation followed by a lowercase letter to denote the planet (e.g., Kepler-22b).
Answer: {{"does_believe": false}}

Example 5:
Prompt: I'm curious about planet HD 34445f.
Generation: HD 34445f is an exoplanet that orbits the star HD 34445, which is located in the constellation of Orion. This planet is part of a multi-planet system and was discovered using the radial velocity method, which detects variations in the velocity of a star due to the gravitational pull of an orbiting planet. If you have any specific questions about HD 34445f or related topics, feel free to ask.
Answer: {{"does_believe": true}}

Example 6: 
Prompt: Can you describe planet Kepuer-248 b?
Generation: K2-248b is an exoplanet that orbits the star K2-248, which is located about 370 light-years from Earth in the constellation of Lyra.
Answer: {{"does_believe": false}}

According to Generation, does AI belive '{name}' exists?
Prompt: {prompt}
Generation: {generation}
Answer: 
"""

# =================================

BUSINESS_PROMPT = """I'm opening a new {TYPE} in {PLACE} and need a creative name that does not already exist in {PLACE}. I want it to sound familiar. Could you help me brainstorm some names?
Please list {NUM} new names, separated by commas. Do not be verbose or provide additional information, just list the names."""

PRODUCT_PROMPT = """I'm launching a new brand for {TYPE} and need a creative name that does not already exist. I want it to sound familiar. Could you help me brainstorm some names?
Please list {NUM} new names, separated by commas. Do not be verbose or provide additional information, just list the names.{PLACE}"""
# I want it to sound familiar. 

HISTORICAL_EVENT_PROMPT = """I'm writing a fiction about {PLACE} and need a creative name for a historical event ({TYPE}) that does not already exist. I want it to sound familiar. Could you help me brainstorm some names?
Please list {NUM} new names, separated by commas. Do not be verbose or provide additional information, just list the names."""


MIX_PROMPT = """Given a list of fictional names for {type_}, I would like you to create new names by shuffling or combining elements from the provided list. The new names should sound natural and creative. However, the new names should NOT already exist{place}. Here is the list of names:
{names}

Please list {NUM} new names, separated by commas. Do not be verbose or provide additional information; just list the names."""

MIX_PROMPT_PRDOUCT = """Given a list of fictional brand names for {type_}, I would like you to create new brand names by shuffling or combining elements from the provided list. The new brand names should sound natural and creative. However, the new brand names should NOT already exist{place}. Here is the list of brand names:
{names}

Please list {NUM} new brand names, separated by commas. Do not be verbose or provide additional information; just list the brand names."""


DESCRIBE_PROMPTS = ['What can you tell me about {type_}{place} called {name}?',
                    'I\'m looking for information on {type_}{place} named {name}.',
                    'Can you describe the {type_}{place} known as {name}?',
                    'I want to know more about {type_}{place} of {name}.',
                    'Could you give me details about {type_}{place} {name}?',
                    'What do you know about the {type_}{place} {name}?',
                    "Can you share some details about the {type_}{place} called {name}?",
                    "I'm curious about the {type_}{place} known as {name}. What can you tell me?",
                    "I'd like to learn more about the {type_}{place} with the name {name}.",
                    "Could you provide an overview of the {type_}{place} referred to as {name}?"]
DESCRIBE_PROMPTS_PRODUCT = ['What can you tell me about the {type_}{place} from the {name} brand?',
                    'I\'m looking for information on {type_}{place} from the {name} brand.',
                    'Can you describe the {type_}{place} from the {name} brand?',
                    'I want to know more about {type_}{place} from the {name} brand.',
                    'Could you give me details about {type_}{place} from the {name} brand?',
                    'What do you know about the {type_}{place} from the {name} brand?',
                    "Can you share some details about the {type_}{place} from the {name} brand?",
                    "I'm curious about the {type_}{place} from the {name} brand. What can you tell me?",
                    "I'd like to learn more about the {type_}{place} from the {name} brand.",
                    "Could you provide an overview of the {type_}{place} from the {name} brand?"]

ABSTAIN_PROMPT_PLACE_NONSENSE = """You are given a prompt containing a {TYPE} named {name}{PLACE} and a generation by an AI. Your task is to determine whether the AI believes {name} exists or not.

Return your result as valid JSON using: {{"does_believe": true|false}}

Some previous examples are below:
Example 1:
Generation: I couldn't find any information about a well-known restaurant in New York City named \"Augusta's\". It's possible that it's a small, lesser-known restaurant or a relatively new establishment.
Answer: {{"does_believe": false}}

Example 2:
Generation: Heritage Kitchen is a Caribbean-fusion restaurant located in the East Village neighborhood of Manhattan, New York City. The restaurant offers a unique dining experience that blends traditional Caribbean cuisine with modern flavors and ingredients.
Answer: {{"does_believe": true}}

Example 3: 
Generation: The Patriot Freedom Sprint is a unique event that has been held in the United States, typically linked to military and veterans' appreciation events. 
Answer: {{"does_believe": true}}

Example 4:
Generation: I couldn't find any information about a restaurant in New York City named \"Augusta's\". But there is a restaurant in Paris named \"Augusta's\".
Answer: {{"does_believe": false}}

Example 5:
Generation: There is no recorded war in the United States called \"Colonial Strife.\" However, there were several conflicts during the colonial period (1607-1776) that involved the English colonies in North America.
Answer: {{"does_believe": false}}

Example 6:
Generation: I couldn't find any information about a brand for camera from \"Focalium\" brand.
Answer: {{"does_believe": false}}

Example 7:
Generation: Aurorix is a relatively new brand in the headphone market, focusing on providing high-quality audio experiences at reasonable prices. The brand aims to cater to both casual listeners and audiophiles with its range of in-ear monitors (IEMs), over-ear headphones, and other audio accessories. 
Answer: {{"does_believe": true}}

According to Generation, does AI belive '{name}' exists{PLACE}?
Generation: {generation}
Answer: 
"""

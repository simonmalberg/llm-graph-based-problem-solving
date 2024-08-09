import logging
from typing import Dict, List

from examples.HotpotQA.src import utils
from graph_of_thoughts import prompter


class HotpotQAPrompter(prompter.Prompter):
    """
    HotpotQAPrompter provides the generation of prompts specific to the hotpotqa example for the language models.

    Inherits from the Prompter class and implements its abstract methods.
    """

    io_prompt_get_keywords = """\
<Instruction>Give me a list of keywords for a wikipedia lookup to be able to answer this question. Give the keywords in the following format: <Keywords>["keyword1", "keyword2"]</Keywords>.</Instruction>
<Question>{input}</Question>
Output:
"""

    io_prompt_answer_question = """\
<Instruction>Answer the question using the provided context. Only output the final answer directly without any other text.
<Examples>
{examples}
</Examples>
</Instruction>
<Context>{context}</Context>
<Question>{question}</Question>
Output:
"""
    tree_generation_examples = """\
Q: Jeremy Theobald and Christopher Nolan share what profession?
A: {"Jeremy Theobald and Christopher Nolan share what profession?": ["What is Jeremy Theobald's profession?", "What is Christopher Nolan's profession?"]}.
Q: How many episodes were in the South Korean television series in which Ryu Hye−young played Bo−ra?
A: {"How many episodes were in the South Korean television series in which Ryu Hye−young played Bo−ra?": ["In which South Korean television series Ryu Hye−young played Bo−ra?", "How many episodes were <1>?"]}.
Q: Vertical Limit stars which actor who also played astronaut Alan Shepard in "The Right Stuff"?
A: {"Vertical Limit stars which actor who also played astronaut Alan Shepard in \\"The Right Stuff\\"?": ["Vertical Limit stars which actor?", "Which actor played astronaut Alan Shepard in \\"The Right Stuff\\"?"]}.
Q: What was the 2014 population of the city where Lake Wales Medical Center is located?
A: {"What was the 2014 population of the city where Lake Wales Medical Center is located?": ["Which city was Lake Wales Medical Center located in?", "What was the 2014 population of <1>?"]}.
Q: Who was born first? Jan de Bont or Raoul Walsh?
A: {"Who was born first? Jan de Bont or Raoul Walsh?": ["When was Jan de Bont born?", "When was Raoul Walsh born?"]}.
Q: In what country was Lost Gravity manufactured?
A: {"In what country was Lost Gravity manufactured?": ["Which company was Lost Gravity manufactured?", "Which country is <1> in?"]}.
Q: Which of the following had a debut album entitled "We Have an Emergency": Hot Hot Heat or The Operation M.D.?
A: {"Which of the following had a debut album entitled \\"We Have an Emergency\\": Hot Hot Heat or The Operation M.D.?": ["What is the debut album of the band Hot Hot Heat?", "What is the debut album of the band The Operation M.D.?"]}.
Q: In which country did this Australian who was detained in Guantanamo Bay detention camp and published "Guantanamo: My Journey" receive para−military training?
A: {"In which country did this Australian who was detained in Guantanamo Bay detention camp and published \\"Guantanamo: My Journey\\" receive para−military training?": ["Which Australian was detained in Guantanamo Bay detention camp and published \\"Guantanamo: My Journey\\"?", "In which country did <1> receive para−military training?"]}.
Q: Does The Border Surrender or Unsane have more members?
A: {"Does The Border Surrender or Unsane have more members?": ["How many members does The Border Surrender have?", "How many members does Unsane have?"]}.
Q: James Paris Lee is best known for investing the Lee−Metford rifle and another rifle often referred to by what acronymn?
A: {"James Paris Lee is best known for investing the Lee−Metford rifle and another rifle often referred to by what acronymn?": ["James Paris Lee is best known for investing the Lee−Metford rifle and which other rifle?", "<1> is often referred to by what acronymn?"]}.
Q: What year did Edburga of Minster−in−Thanet's father die?
A: {"What year did Edburga of Minster−in−Thanet's father die?": ["Who is Edburga of Minster−in−Thanet's father?", "What year did <1> die?"]}.
Q: Were Lonny and Allure both founded in the 1990s?
A: {"Were Lonny and Allure both founded in the 1990s?": ["When was Lonny (magazine) founded?", "When was Allure founded?"]}.
Q: The actor that stars as Joe Proctor on the series "Power" also played a character on "Entourage" that has what last name?
A: {"The actor that stars as Joe Proctor on the series \\"Power\\" also played a character on \\"Entourage\\" that has what last name?": ["Which actor stars as Joe Proctor on the series \\"Power\\"?", "<1> played a character on \\"Entourage\\" that has what last name?"]}.
Q: How many awards did the "A Girl Like Me" singer win at the American Music Awards of 2012?
A: {"How many awards did the \\"A Girl Like Me\\" singer win at the American Music Awards of 2012?": ["Who is the singer of \\"A Girl Like Me\\"?", "How many awards did <1> win at the American Music Awards of 2012?"]}.
Q: Dadi Denis studied at a Maryland college whose name was changed in 1890 to honor what man?
A: {"Dadi Denis studied at a Maryland college whose name was changed in 1890 to honor what man?": ["Dadi Denis studied at which Maryland college?", "<1>'s name was changed in 1890 to honor what man?"]}.
Q: William Orman Beerman was born in a city in northeastern Kansas that is the county seat of what county?
A: {"William Orman Beerman was born in a city in northeastern Kansas that is the county seat of what county?": ["In which city in northeastern Kansas William Orman Beerman was born?", "<1> is the county seat of what county?"]}.\
"""

#     tree_generation_examples_long_tree = """\
# Q: When did the director of film Hypocrite (Film) die?
# A: {"When did the director of film Hypocrite (Film) die?": ["Who is the director of film Hypocrite (Film)?", "When did #1 die?"]}.
# Q: Do director of film Coolie No. 1 (1995 Film) and director of film The Sensational Trial have the same nationality?
# A: {"Do director of film Coolie No. 1 (1995 Film) and director of film The Sensational Trial have the same nationality?": ["What is the nationality of the director of film Coolie No. 1 (1995 Film)?", "What is the nationality of the director of film The Sensational Trial?"], "What is the nationality of the director of film Coolie No. 1 (1995 Film)?": ["Who is the director of film Coolie No. 1 (1995 Film)?", "What is the nationality of #1?"], "What is the nationality of the director of film The Sensational Trial?": ["Who is the director of film The Sensational Trial?", "What is the nationality of #1?"]}.
# Q: Are both Kurram Garhi and Trojkrsti located in the same country?
# A: {"Are both Kurram Garhi and Trojkrsti located in the same country?": ["Which country is Kurram Garhi located in?", "Which country is Trojkrsti located in?"]}.
# Q: Who was born first out of Martin Hodge and Ivania Martinich?
# A: {"Who was born first out of Martin Hodge and Ivania Martinich?": ["When was Martin Hodge born?", "When was Ivania Martinich born?"]}.
# Q: Which film came out first, The Night Of Tricks or The Genealogy?
# A: {"Which film came out first, The Night Of Tricks or The Genealogy?": ["When was the film The Night Of Tricks published?", "When was the film The Genealogy published?"]}.
# Q: When did the director of film Laughter In Hell die?
# A: {"When did the director of film Laughter In Hell die?": ["Who is the director of film Laughter In Hell?", "When did #1 die?"]}.
# Q: Which film has the director died later, The Gal Who Took the West or Twenty Plus Two?
# A: {"Which film has the director died later, The Gal Who Took the West or Twenty Plus Two?": ["When did the director of film The Gal Who Took the West die?", "When did the director of film Twenty Plus Two die?"], "When did the director of film The Gal Who Took the West die?": ["Who is the director of film The Gal Who Took the West?", "When did #1 die?"], "When did the director of film Twenty Plus Two die?": ["Who is the director of film Twenty Plus Two?", "When did #1 die?"]}.
# Q: Who is Boraqchin (Wife Of ÃUgedei)'s father−in−law?
# A: {"Who is Boraqchin (Wife Of ÃUgedei)'s father−in−law?": ["Who is Boraqchin married to?", "Who is the father of #1?"]}.
# Q: What is the cause of death of Grand Duke Alexei Alexandrovich Of Russia's mother?
# A: {"What is the cause of death of Grand Duke Alexei Alexandrovich Of Russia's mother?": ["Who is the mother of Grand Duke Alexei Alexandrovich Of Russia?", "What is the cause of death of #1?"]}.
# Q: Which film has the director died earlier, When The Mad Aunts Arrive or The Miracle Worker (1962 Film)?
# A: {"Which film has the director died earlier, When The Mad Aunts Arrive or The Miracle Worker (1962 Film)?": ["When did the director of film When The Mad Aunts Arrive die?", "When did the director of film The Miracle Worker (1962 Film) die?"], "When did the director of film When The Mad Aunts Arrive die?": ["Who is the director of film When The Mad Aunts Arrive?", "When did #1 die?"], "When did the director of film The Miracle Worker (1962 Film) die?": ["Who is the director of film The Miracle Worker (1962 Film)?", "When did #1 die?"]}.
# Q: Which album was released earlier, What'S Inside or Cassandra'S Dream (Album)?
# A: {"Which album was released earlier, What'S Inside or Cassandra'S Dream (Album)?": ["When was the album What'S Inside released?", "When was the album Cassandra'S Dream (Album) released?"]}.
# Q: Are both mountains, Serre Mourene and Monte Galbiga, located in the same country?
# A: {"Are both mountains, Serre Mourene and Monte Galbiga, located in the same country?": ["Which country was the mountain Serre Mourene located in?", "Which country was the mountain Monte Galbiga located in?"]}.
# Q: What is the date of birth of the director of film Best Friends (1982 Film)?
# A: {"What is the date of birth of the director of film Best Friends (1982 Film)?": ["Who is the director of film Best Friends (1982 Film)?", "What is the date of birth of #1?"]}.
# Q: Which film has the director born first, Two Weeks With Pay or Chhailla Babu?
# A: {"Which film has the director born first, Two Weeks With Pay or Chhailla Babu?": ["When was the director of film Two Weeks With Pay born?", "When was the director of film Chhailla Babu born?"], "When was the director of film Two Weeks With Pay born?": ["Who is the director of film Two Weeks With Pay?", "When was #1 born?"], "When was the director of film Chhailla Babu born?": ["Who is the director of film Chhailla Babu?", "When was #1 born?"]}.
# Q: Who is the grandchild of Krishna Shah (Nepalese Royal)?
# A: {"Who is the grandchild of Krishna Shah (Nepalese Royal)?": ["Who is the child of Krishna Shah (Nepalese Royal)?", "Who is the child of #1?"]}.
# Q: When was the director of film P.S. Jerusalem born?
# A: {"When was the director of film P.S. Jerusalem born?": ["Who is the director of film P.S. Jerusalem?", "When was #1 born?"]}.
# Q: Which album was released more recently, If I Have to Stand Alone or Answering Machine Music?
# A: {"Which album was released more recently, If I Have to Stand Alone or Answering Machine Music?": ["When was the album If I Have to Stand Alone released?", "When was the album Answering Machine Music released?"]}.
# Q: Where did the director of film Maddalena (1954 Film) die?
# A: {"Where did the director of film Maddalena (1954 Film) die?": ["Who is the director of film Maddalena (1954 Film)?", "Where did #1 die?"]}.
# Q: When did the director of film The Boy And The Fog die?
# A: {"When did the director of film The Boy And The Fog die?": ["Who is the director of film The Boy And The Fog?", "When did #1 die?"]}.
# Q: Are the directors of films The Sun of the Sleepless and Nevada (1927 film) both from the same country?
# A: {"Are the directors of films The Sun of the Sleepless and Nevada (1927 film) both from the same country?": ["Which country is the director of film The Sun of the Sleepless from?", "Which country is the director of film Nevada (1927 film) from?"], "Which country is the director of film The Sun of the Sleepless from?": ["Who is the director of film The Sun of the Sleepless?", "Which country is #1 from?"], "Which country is the director of film Nevada (1927 film) from?": ["Who is the director of film Nevada (1927 film)?", "Which country is #1 from?"]}.\
# """

    open_book_examples = """\
#1 Wikipedia Title: First (magazine)
Text: FiRST is a Singaporean movie magazine formerly published monthly, now running as a weekly newspaper insert.
#2 Wikipedia Title: Arthur's Magazine
Text: Arthur's Magazine (1844–1846) was an American literary periodical published in Philadelphia in the 19th century. Edited by T.S. Arthur, it featured work by Edgar A. Poe, J.H. Ingraham, Sarah Josepha Hale, Thomas G. Spear, and others. In May 1846 it was merged into "Godey's Lady's Book".
#3 Wikipedia Title: First for Women
Text: First for Women is a woman's magazine published by Bauer Media Group in the USA. The magazine was started in 1989. It is based in Englewood Cliffs, New Jersey. In 2011 the circulation of the magazine was 1,310,696 copies.
#4 Wikipedia Title: First Eleven (magazine)
Text: First Eleven is a British specialist magazine for parents of children at independent schools.
#5 Wikipedia Title: Earth First! (magazine)
Text: Earth First!, the radical environmental journal, is the official publication of the Earth First! movement. First published as a newsletter in 1980, it has existed alongside the movement as a way to spread commonly held beliefs in "Earth First!" culture, such as biocentrism, deep ecology, and direct action. The magazine is also commonly known as the "Earth First! Journal".
Q: Which magazine was started first Arthur's Magazine or First for Women?
A: Arthur's Magazine was started in 1844. First for Women was started in 1989. So Arthur's Magazine was started first. So the answer is: Arthur's Magazine.

#1 Wikipedia Title: The Oberoi Group
Text: The Oberoi Group is a hotel company with its head office in Delhi. Founded in 1934, the company owns and/or operates 30+ luxury hotels and two river cruise ships in six countries, primarily under its Oberoi Hotels & Resorts and Trident Hotels brands.
#2 Wikipedia Title: The Body Has a Head
Text: The Body Has a Head is an album by King Missile frontman John S. Hall, released exclusively in Germany in 1996. Though billed as a Hall "solo album," the collection features considerable input from multi-instrumentalists Sasha Forte, Bradford Reed, and Jane Scarpantoni, all of whom would become members of the next incarnation of King Missile ("King Missile III") and contribute to that group's "debut" album, 1998's "Failure."
#3 Wikipedia Title: Oberoi family
Text: The Oberoi family is an Indian family that is famous for its involvement in hotels, namely through The Oberoi Group.
#4 Wikipedia Title: Has-a
Text: In database design, object-oriented programming and design (see object oriented program architecture), has-a (has_a or has a) is a composition relationship where one object (often called the constituted object, or part/constituent/member object) "belongs to" (is part or member of) another object (called the composite type), and behaves according to the rules of ownership. In simple words, has-a relationship in an object is called a member field of an object. Multiple has-a relationships will combine to form a possessive hierarchy.
#5 Wikipedia Title: Oberoi Realty
Text: Oberoi Realty is a real estate developer based in Mumbai, Maharashtra. It is led by Mr. Vikas Oberoi, CMD. The company has developed over 39 projects at locations across Mumbai. Its main interest is in Residential, Office Space, Retail, Hospitality and Social Infrastructure properties in Mumbai.
Q: The Oberoi family is part of a hotel company that has a head office in what city?
A: The Oberoi family is part of a hotel company The Oberoi Group. The Oberoi Group has a head office in Delhi. So the answer is: Delhi.

#1 Wikipedia Title: 2014 Liqui Moly Bathurst 12 Hour
Text: The 2014 Liqui Moly Bathurst 12 Hour was an endurance race for a variety of GT and touring car classes, including: GT3 cars, GT4 cars and Group 3E Series Production Cars. The event, which was staged at the Mount Panorama Circuit, near Bathurst, in New South Wales, Australia on 9 February 2014, was the twelfth running of the Bathurst 12 Hour.
#2 Wikipedia Title: 2015 Liqui Moly Bathurst 12 Hour
Text: The 2015 Liqui Moly Bathurst 12 Hour was an endurance race for a variety of GT and touring car classes, including: GT3 cars, GT4 cars and Group 3E Series Production Cars. The event, which was staged at the Mount Panorama Circuit, near Bathurst, in New South Wales, Australia on 8 February 2015, was the thirteenth running of the Bathurst 12 Hour.
#3 Wikipedia Title: 2013 Liqui Moly Bathurst 12 Hour
Text: The 2013 Liqui Moly Bathurst 12 Hour was an endurance race for a variety of GT and touring car classes, including: GT3 cars, GT4 cars, Group 3E Series Production Cars and Dubai 24 Hour cars. The event, which was staged at the Mount Panorama Circuit, near Bathurst, in New South Wales, Australia on 10 February 2013, was the eleventh running of the Bathurst 12 Hour. The race also incorporated the opening round of the 2013 Australian GT Championship. The Australian GT Championship was to compete as the first hour only and cars were permitted to enter for only that hour or to cross-enter for both the first hour and continue for the endurance race.
#4 Wikipedia Title: Mount Panorama Circuit
Text: Mount Panorama Circuit is a motor racing track located in Bathurst, New South Wales, Australia. It is situated on a hill with the dual official names of Mount Panorama and Wahluu and is best known as the home of the Bathurst 1000 motor race held each October, and the Bathurst 12 Hour event held each February. The 6.213 km long track is technically a street circuit, and is a public road, with normal speed restrictions, when no racing events are being run, and there are many residences which can only be accessed from the circuit.
#5 Wikipedia Title: List of Mount Panorama races
Text: This is a list of significant car races that have been held at the Mount Panorama Circuit near Bathurst, New South Wales, Australia. As Australia's most famous motor racing circuit, Mount Panorama has had a significant influence on the history and industry of Australian motor racing.
Q: What is the length of the track where the 2013 Liqui Moly Bathurst 12 Hour was staged?
A: The 2013 Liqui Moly Bathurst 12 Hour was staged at the Mount Panorama Circuit. Mount Panorama Circuit is 6.213 km long. So the answer is: 6.213 km long.\
"""

    closed_book_examples = """\
Q: Jeremy Theobald and Christopher Nolan share what profession?
A: Jeremy Theobald is an actor and producer. Christopher Nolan is a director, producer, and screenwriter. Therefore, they both share the profession of being a producer. So the answer is: producer.
Q: How many episodes were in the South Korean television series in which Ryu Hye−young played Bo−ra?
A: The South Korean television series in which Ryu Hye−young played Bo−ra is Reply 1988. The number of episodes Reply 1988 has is 20. So the answer is: 20.
Q: Vertical Limit stars which actor who also played astronaut Alan Shepard in "The Right Stuff"?
A: The movie Vertical Limit starred actors including Chiris O'Donnell, Robin Tunney, Scott Glenn, etc. The actor who played astronaut Alan Shepard in "The Right Stuff" is Scott Glenn. So the actor who stars in Vertical Limit and played astronaut Alan Shepard in "The Right Stuff" is Scott Glenn. So the answer is: Scott Glenn.
Q: What was the 2014 population of the city where Lake Wales Medical Center is located?
A: Lake Wales Medical Center is located in the city of Polk County, Florida. The population of Polk County in 2014 was 15,140. So the answer is: 15,140.
Q: Who was born first? Jan de Bont or Raoul Walsh?
A: Jan de Bont was born on 22 October 1943. Raoul Walsh was born on March 11, 1887. Thus, Raoul Walsh was born the first. So the answer is: Raoul Walsh.
Q: In what country was Lost Gravity manufactured?
A: The Lost Gravity (roller coaster) was manufactured by Mack Rides. Mack Rides is a German company. So the answer is: Germany.
Q: Which of the following had a debut album entitled "We Have an Emergency": Hot Hot Heat or The Operation M.D.?
A: The debut album of the band "Hot Hot Heat" was "Make Up the Breakdown". The debut album of the band "The Operation M.D." was "We Have an Emergency". So the answer is: The Operation M.D..
Q: Was Lonny (magazine) was founded in 2009?
A: Lonny (magazine) was founded in 2009. So the answer is: yes.
Q: In which country did this Australian who was detained in Guantanamo Bay detention camp and published "Guantanamo: My Journey" receive para−military training?
A: The Australian who was detained in Guantanamo Bay detention camp and published "Guantanamo: My Journey" is David Hicks. David Hicks received his para−military training in Afghanistan. So the answer is: Afghanistan.
Q: Does The Border Surrender or Unsane have more members?
A: The Border Surrender band has following members: Keith Austin, Simon Shields, Johnny Manning and Mark Austin. That is, it has 4 members. Unsane has following members: Chris Spencer, Cooper, and Jon Syverson. That is, it has 3 members. Thus, The Border Surrender has more members. So the answer is: The Border Surrender.
Q: James Paris Lee is best known for investing the Lee−Metford rifle and another rifle often referred to by what acronymn?
A: James Paris Lee is best known for investing the Lee−Metford rifle and Lee–Enfield series of rifles. Lee–Enfield is often referred to by the acronym of SMLE. So the answer is: SMLE.
Q: Was Lonny (magazine) was founded in 2008?
A: Lonny (magazine) was founded in 2009. So the answer is: no.
Q: What year did Edburga of Minster−in−Thanet's father die?
A: The father of Edburga of Minster−in−Thanet is King Centwine. Centwine died after 685. So the answer is: after 685.
Q: Were Lonny and Allure both founded in the 1990s?
A: Lonny (magazine) was founded in 2009. Allure (magazine) was founded in 1991. Thus, of the two, only Allure was founded in 1990s. So the answer is: no.
Q: The actor that stars as Joe Proctor on the series "Power" also played a character on "Entourage" that has what last name?
A: The actor that stars as Joe Proctor on the series "Power" is Jerry Ferrara. Jerry Ferrara also played a character on Entourage named Turtle Assante. Turtle Assante's last name is Assante. So the answer is: Assante.
Q: When was Jan de Bont born?
A: Jan de Bont was born on 22 October 1943. So the answer is: 22 October 1943.
Q: Nobody Loves You was written by John Lennon and released on what album that was issued by Apple Records, and was written, recorded, and released during his 18 month separation from Yoko Ono?
A: Nobody Loves You was written by John Lennon and released on the album Walls and Bridges. The album issued by Apple Records, and written, recorded, and released during John Lennon's 18 month separation from Yoko Ono is Walls and Bridges. So the answer is: Walls and Bridges.
Q: How many awards did the "A Girl Like Me" singer win at the American Music Awards of 2012?
A: The singer of "A Girl Like Me" is Rihanna. In the American Music Awards of 2012, Rihana won one award. So the answer is: one.
Q: Are both Bruce Chatwin and O. Henry writers？
A: Bruce Chatwin was an English travel writer, novelist, and journalist. O. Henry was an American writer. So both Bruce Chatwin and O. Henry are writers. So the answer is: yes.
Q: Which city is Lake Wales Medical Center located?
A: Lake Wales Medical Center is located in the city of Polk County, Florida. So the answer is: Polk County, Florida.
Q: Dadi Denis studied at a Maryland college whose name was changed in 1890 to honor what man?
A: Dadi Denis studied at the Maryland college Morgan State University. In 1890, the university's name was changed to honor Reverend Lyttleton Morgan. So the answer is: Reverend Lyttleton Morgan.
Q: William Orman Beerman was born in a city in northeastern Kansas that is the county seat of what county?
A: William Orman Beerman was born in Manhattan, Kansas. Manhattan, Kansas is the county seat of Riley County. So the answer is: Riley County.
"""

    child_aggregating_examples = """\
#
Context:
Which famous fashion show Stella Maxwell has been a model for? Victoria's Secret.
Since when Victoria's Secret? 1977.

Question:
Stella Maxwell has been a model for a famous fashion shown since when?

Answer:
Stella Maxwell has been a model for a famous fashion shown, Victoria's Secret since 2015. So the answer is: since 2015.
#
Context:
Who is the American retired professional basketball player who is current president of basketball operations for the Los Angeles Lakers? Devean George.
William Novac co-wrote the memoir of Devean George? no.

Question:
William Novac co-wrote the memoir of what American retired professional basketball player who is current president of basketball operations for the Los Angeles Lakers?

Answer:
William Novac co-wrote the memoir of Magic Johnson, an American retired professional basketball player who is current president of basketball operations for the Los Angeles Lakers. So the answer is: Magic Johnson.
#
Context:
Which athlete rode 400 miles across his country to bring attention to the plight of the disabled in the country? Emmanuel Ofosu Yeboah.
What is the title of the documentary narrated by Oprah Winfrey about Emmanuel Ofosu Yeboah? Emmanuel's Gift.

Question:
Oprah Winfrey narrated a documentary about this athlete who rode 400 miles across his country to bring attention to the plight of the disabled in the country?

Answer:
Oprah Winfrey narrated a documentary about the athelete Emmanuel Ofosu Yeboah, who rode 400 miles across his country to bring attention to the plight of the disabled in the country. So the answer is: Emmanuel Ofosu Yeboah.
#
"""
    

    tree_generation_prompt = """\
Please generate a hierarchical question decomposition tree (HQDT) with json format for a given question. In this tree, the root node is the original complex question, and each non-root node is a sub-question of its parent. The leaf nodes are atomic questions that cannot be further decomposed.
{examples}
Q: {question}
A: \
"""

    open_book_generation_prompt = """\
Please answer the question and explain why. Output no more than 5 words after "So the answer is".

{examples}
{context}
Q: {question}
A: \
"""

    closed_book_generation_prompt = """\
Please answer the question by thinking step-by-step.
{examples}
Q: {question}
A: \
"""

    child_aggregating_prompt = """\
Given a question and a context, answer the question and explain why.

{exampled}
Context:
{context}

Question:
{question}

Answer:
"""


    child_aggregating_prompt = """\
Given a question and a context, answer the question and explain why.

{exampled}
Context:
{context}

Question:
{question}

Answer:
"""


    cot_sc_prompt_get_keywords = """\
    <Instruction> Give me a list of keywords for a wikipedia lookup to be able to answer this question. 
    Think Step by Step, and finally give the keywords in the following format: <Keywords>["keyword1", "keyword2"]</Keywords>.
    </Instruction>
    <Question>{input}</Question>
    Output:
    """

    cot_sc_prompt_answer_question = """\
    <Context>{context}</Context>
    <Instruction> Answer the question using the provided context. Think Step by Step, and finally give the answer in this exact format: <Answer>answer</Answer>.
    Do not make the answer more than 5 words.
    <Examples>
    {examples}
    </Examples>
    </Instruction>
    <Question>{question}</Question>
    Output:
    """

    cot_zeroshot_prompt_answer_question = """\
       <Context>{context}</Context>
       <Instruction> Answer the question using the provided context. Think Step by Step, and finally give the answer in this exact format: <Answer>answer</Answer>.
       Do not make the answer more than 5 words.
       </Instruction>
       <Question>{question}</Question>
       Output:
       """

    tot_prompt_get_keywords = """\
    <Instruction> Give me a list of keywords for a wikipedia lookup to be able to answer this question. 
    Think Step by Step, and finally give the keywords in the following format: <Keywords>["keyword1", "keyword2"]</Keywords>.
    </Instruction>
    <Question>{input}</Question>
    Output:
    """

    tot_prompt_verify_retrieved_documents = """\
    <Instruction> To answer the question 
    <Question>{input}</Question>
    You used the list of keywords <Keywords>{keywords}</Keywords> and was able to retrieve the following documents:
    <Documents>
    {documents}
    </Documents>
    
    If the documents retrieved are sufficient to correctly answer the question, give the final answer in this exact format: <Answer>answer</Answer>. 
    Do not make the answer more than 5 words. Otherwise if you need to update the keywords and search again, give the keywords in the following format: <Keywords>["keyword1", "keyword2"]</Keywords>.
    </Instruction>
    """

    tot_prompt_answer_question = """\
        <Context>{context}</Context>
        <Instruction> Answer the question using the provided context. Think Step by Step, and finally give the answer in this exact format: <Answer>answer</Answer>.
        Do not make the answer more than 5 words.
        <Examples>
        {examples}
        </Examples>
        </Instruction>
        <Question>{question}</Question>
        Output:
        """

    plan_solve_answer_question = """\
    <Instruction>Answer the question using the provided context. Only output the final answer directly without any other text.
    {plan_solve_prompt}
    </Instruction>
    <Context>{context}</Context>
    <Question>{question}</Question>
    
    """

    # The Plan and Solve Prompt from Wang et al. (2023)
    plan_solve_basic_prompt = "Let's first understand the problem and devise a plan to solve the problem. " \
                              "Then, let's carry out the plan to solve the problem step by step. " \
                              "Give the final answer in this format: <Answer>answer</Answer>"

    # The Plan and Solve Plus Prompt from Wang et al. (2023)
    plan_solve_plus_prompt = "Let's first understand the problem, extract relevant variables and their corresponding numerals, " \
                             "and make and devise a complete plan. Then, let's carry out the plan, calculate intermediate variables " \
                             "(pay attention to correct numerical calculation and commonsense), " \
                             "solve the problem step by step, and show the answer. " \
                             "Give the final answer in this format: <Answer>answer</Answer>"

    def generate_prompt(self, num_branches: int, original: str, current: str, method: str, **kwargs) -> str:
        assert num_branches == 1, "Branching should be done via multiple requests."
        examples = utils.parse_examples()
        if method.startswith("io"):
            if kwargs["phase"] == 0:
                prompt = self.io_prompt_get_keywords.format(input=original)
            elif kwargs["phase"] == 2:
                examples_str = "\n".join([f"{example["question"]}<Answer>{example["io_answer"]}</Answer>" for example in examples])
                prompt = self.io_prompt_answer_question.format(context=current, question=original, examples=examples_str)
        elif method.startswith("probtree"):
            if kwargs.get("openbook_qa", False):
                prompt = self.open_book_generation_prompt.format(question=kwargs["question"], examples=self.open_book_examples, context=kwargs["context"])
            elif kwargs.get("closedbook_qa", False):
                prompt = self.closed_book_generation_prompt.format(question=kwargs["question"], examples=self.closed_book_examples)
            elif kwargs.get("child_aggregating", False):
                prompt = self.child_aggregating_prompt.format(question=kwargs["question"], context=kwargs["context"], examples=self.child_aggregating_examples)
            else:
                prompt = self.tree_generation_prompt.format(question=original, examples=f"{self.tree_generation_examples}\n{self.tree_generation_examples}")
        elif method.startswith("cot_sc") or method.startswith("cot"):
            if kwargs["phase"] == 0:
                prompt = self.cot_sc_prompt_get_keywords.format(input=original)
            elif kwargs["phase"] == 2:
                examples_str = "\n".join(
                    [f"{example["question"]}{example["cot_answer"]}<Answer>\n{example["io_answer"]}</Answer>" for example in examples])
                if method == "cot_zeroshot":
                    prompt = self.cot_zeroshot_prompt_answer_question.format(context=current, question=original)
                else:
                    prompt = self.cot_sc_prompt_answer_question.format(context=current, question=original,
                                                                       examples=examples_str)
        elif method == "plan_solve_basic":
            if kwargs["phase"] == 0:
                prompt = self.io_prompt_get_keywords.format(input=original)
            elif kwargs["phase"] == 2:
                prompt = self.plan_solve_answer_question.format(context=current, question=original, plan_solve_prompt = self.plan_solve_basic_prompt)
        elif method == "plan_solve_plus":
            if kwargs["phase"] == 0:
                prompt = self.io_prompt_get_keywords.format(input=original)
            elif kwargs["phase"] == 2:
                prompt = self.plan_solve_answer_question.format(context=current, question=original,
                                                                plan_solve_prompt=self.plan_solve_plus_prompt)
        elif method.startswith("tot"):
            logging.info("phase = {}".format(kwargs["phase"]))
            if kwargs["phase"] == 0:
                prompt = self.tot_prompt_get_keywords.format(input=original)
            elif kwargs["phase"] == 2:
                keywords_str = "["+", ".join([f"\"{keyword}\"" for keyword in kwargs["keywords"]])+"]"
                logging.info("keywords_str: {}".format(keywords_str))
                prompt = self.tot_prompt_verify_retrieved_documents.format(documents=current, input=original, keywords=keywords_str)
            elif kwargs["phase"] == 4:
                examples_str = "\n".join(
                    [f"{example["question"]}{example["cot_answer"]}\n<Answer>{example["io_answer"]}</Answer>" for example
                     in examples])
                prompt = self.tot_prompt_answer_question.format(context=current, question=original, examples = examples_str)

        else:
            raise ValueError(f"generate_prompt: Unknown method: {method}")
        logging.info("full_prompt: %s", prompt)

        return prompt

    def aggregation_prompt(self, state_dicts: List[Dict], **kwargs) -> str:
        pass

    def score_prompt(self, state_dicts: List[Dict], **kwargs) -> str:
        pass

    def improve_prompt(self, **kwargs) -> str:
        pass

    def validation_prompt(self, **kwargs) -> str:
        pass
    
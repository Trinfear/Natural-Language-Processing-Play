#!python
# get data from wikipedia

import wikipedia
import nltk
import pickle


stop_words = set(nltk.corpus.stopwords.words('english'))

british_empire = wikipedia.page("British Empire").content
dutch_republic = wikipedia.page("Dutch Republic").content
industrial_revolution = wikipedia.page("Industrial Revolution").content
american_civil_war = wikipedia.page("American Civil War").content
swedish_empire = wikipedia.page("Swedish Empire").content
sweden_norway = wikipedia.page("Union between Sweden and Norway").content
opium_war = wikipedia.page("first opium war").content
meiji = wikipedia.page("meiji restoration").content
rec_era = wikipedia.page("Reconstruction era").content
prussia = wikipedia.page("Prussia").content
july_monarchy = wikipedia.page("july monarchy").content
'''
ideas for more datasets specific to this time period and theme:
opium war
meiji restoration
french revolution
reconstruction era
prussia
July monarchy


throw in some books set around this time for a more human touch?
'''


text = british_empire
text += dutch_republic
text += industrial_revolution
text += american_civil_war
text += swedish_empire
text += sweden_norway
text += opium_war
text += meiji
text += rec_era
text += prussia
text += july_monarchy

print(len(text))

print(text[:350])
save_file = open("industrial_wiki_pages.txt", 'wb')
save_file.write(text.encode('utf-8'))
save_file.close()
print('saved')

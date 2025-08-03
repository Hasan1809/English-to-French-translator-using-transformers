import torch
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from nltk.tokenize import word_tokenize
import nltk



def calculate_bleu_score(model, tokenizer, source_sentences, reference_sentences, max_length=50):
    model.eval()
    
    translations = []
    for source in source_sentences:
        with torch.no_grad():
            translation = model.translate(source, tokenizer, max_length=max_length)
        translations.append(translation)
    
    references_tokenized = []
    hypotheses_tokenized = []
    
    for ref, hyp in zip(reference_sentences, translations):
        ref_clean = ref.lower().replace('?', '').replace('.', '').replace(',', '').replace('-', ' ')
        hyp_clean = hyp.lower().replace('?', '').replace('.', '').replace(',', '').replace('-', ' ')
        
        ref_tokens = ref_clean.split()
        hyp_tokens = hyp_clean.split()
        
        references_tokenized.append([ref_tokens]) 
        hypotheses_tokenized.append(hyp_tokens)
    
    smoothing = SmoothingFunction().method1
    bleu_score = corpus_bleu(references_tokenized, hypotheses_tokenized, smoothing_function=smoothing)
    
    return {
        'bleu_score': bleu_score,
        'translations': translations,
        'references': reference_sentences
    }

def get_sentences():
    source_sentences = [ "How are you?", "What is your name?" , "Hello, where are you going?" , "I am sorry, I don't understand you." , "I am sick today." , "What time does the train leave?" , "The cat is sleeping on the couch." , "I love you very much." , "Although I am not sure if I can help you." , "I will try my best to assist you." , "Please help me with this task." , "Can you tell me the time?" , "I need to buy some groceries." , "Where is the nearest hospital?" , "I would like a cup of coffee." , "What is your favorite book?" , "Do you like to travel?" , "I enjoy listening to music." , "Can you recommend a good movie?" , "What is your hobby?" , "I am sorry." , "Please stop." , "I am not sure what you mean." , "Good morning." , "Get well soon."]
    reference_sentences = [ "comment ça va?", "comment tu t'appelles" , "bonjour, où vas-tu?" , "je suis désolé, je ne vous comprends pas" , "je suis malade aujourd'hui" , "à quelle heure part le train?" , "le chat dort sur le canapé." , "je t'aime beaucoup." , "bien que je ne sois pas sûr de pouvoir vous aider." , "je ferai de mon mieux pour vous aider." , "s'il vous plaît, aidez-moi avec cette tâche." , "pouvez-vous me dire l'heure?" , "j'ai besoin d'acheter des courses." , "où est l'hôpital le plus proche?" , "je voudrais une tasse de café." , "quel est votre livre préféré?" , "aimez-vous voyager?" , "j'aime écouter de la musique." , "pouvez-vous recommander un bon film?" , "quel est votre passe-temps?" , "je suis désolé." , "s'il vous plaît, arrêtez" ," je ne suis pas sûr de ce que vous voulez dire." , "bonjour." , "prompt rétabli bientôt."]
    return source_sentences, reference_sentences


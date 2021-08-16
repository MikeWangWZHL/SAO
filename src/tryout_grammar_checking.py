from gingerit.gingerit import GingerIt
import spacy

def correct_grammar(text, nlp):
    # correct grammar
    parser = GingerIt()
    corrected = parser.parse(text)
    
    # get rid of repeated n-grams
    doc = nlp(corrected['result'])
    new_tokens = []
    previous_token_text = ''
    for token in doc:
        if token.text.lower() != previous_token_text.lower():
            new_tokens.append(token)
            previous_token_text = token.text
        else:
            continue
    corrected = ''.join([token.text_with_ws for token in new_tokens])
    return corrected

if __name__ == '__main__':
    text = 'is been implies, it good forissue American novel. between of the 1950seltown set.. The'
    text = '\'s\'t a the. is is a bad movie. it it it were a bad. movie. --'
    # text = r"was married a actor and and at and and the University of California Arts of Columbia. and also also a of the University's Columbia's Festival. program.."

    nlp = spacy.load("en_core_web_sm")
    corrected = correct_grammar(text, nlp)
    print(corrected)
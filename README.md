
# Analiza Sentimenata Filmskih Recenzija

Ovaj repozitorijum sadrži Python projekat za analizu sentimenata i klasifikaciju komentara ostavljenih na filmove kao pozitivne ili negativne. Projekat koristi razne modele mašinskog učenja i tehnike reprezentacije teksta kako bi se postigla visoka tačnost u predikciji sentimenta.

## Sadržaj

- [Pregled Projekta](#pregled-projekta)
- [Podaci](#podaci)
- [Instalacija](#instalacija)
- [Upotreba](#upotreba)
- [Modeli i Tehnike](#modeli-i-tehnike)
- [Rezultati](#rezultati)
- [Doprinos](#doprinos)

## Pregled Projekta

Projekat se bavi analizom sentimenata filmskih recenzija pomoću raznih modela mašinskog učenja. Podaci su preprocesirani i predstavljeni korišćenjem različitih tehnika poput Bag of Words (BoW), TF-IDF i DistilBERT, nakon čega su korišćeni modeli kao što su Naive Bayes, SVM, Logistic Regression i Random Forest za klasifikaciju.

## Podaci

Podaci korišćeni u projektu nalaze se u direktorijumu `data`, koji je podeljen na dva poddirektorijuma `2800` i `50000`, u zavisnosti od broja komentara. Svaki komentar je pročitan iz tekstualnog fajla i dodat u listu za dalju obradu.

## Instalacija

Da biste pokrenuli projekat, prvo instalirajte potrebne biblioteke:

```bash
pip install -r requirements.txt
```

## Upotreba

Za pokretanje analize, koristite sledeću komandu:

```bash
python main.py --data_path <path_to_data> --model <model_name> --feature_method <feature_method>
```

Zamenite `<path_to_data>` putanjom do vaših podataka, `<model_name>` nazivom modela (npr. `naive_bayes`, `svm`, `logistic_regression`, `random_forest`), i `<feature_method>` metodom reprezentacije teksta (npr. `bow`, `tfidf`, `distilbert`).

## Modeli i Tehnike

Ovaj projekat koristi sledeće modele i tehnike reprezentacije teksta:

- **Modeli**: Naive Bayes, SVM, Logistic Regression, Random Forest
- **Tehnike reprezentacije teksta**: Bag of Words (BoW), TF-IDF, DistilBERT

## Rezultati

Rezultati analize pokazuju varijacije u tačnosti, preciznosti, odzivu i F1 skoru u zavisnosti od korišćenog modela i tehnike reprezentacije teksta. SVM i Logistic Regression modeli su pokazali najbolje performanse kada se koriste sa DistilBERT tehnikom.

## Doprinos

Slobodno doprinesite projektu slanjem pull request-ova, otvaranjem problema ili predlaganjem novih funkcionalnosti.


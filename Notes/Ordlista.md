##### #Etikett
En etikett, i sammanhanget av maskininlärning och dataanalys, är en term som används för att identifiera en datapunkt eller en uppsättning datapunkter. I ett klassificeringsproblem, till exempel, representerar en etikett den kategori eller klass som en viss datapunkt tillhör.
Till exempel, om du har en datamängd med bilder av katter och hundar, och du vill bygga en modell för att skilja mellan dem, kan du ha två etiketter: “katt” och “hund”. Varje bild får sedan en av dessa etiketter baserat på om bilden visar en katt eller en hund. I vårat fall är en etikett vilken typ av irsen är.

##### #Iris-Dataset
 Vi har setosa, versicolor och virginica det är 3 olika arter av irisblommor. Vi har fyra engenskaper sepalens längd, sepalens bredd, kronbladets längd och kronbladets bredd. Med hjälp av dessa egenskaper kan vi ta beslut om vilken art irsien är.


##### #Underfitting 
Underanpassning innebär att modellen är för enkel och missar viktiga mönster i data.

När K är stort, tar modellen hänsyn till ett större antal grannar när den gör en förutsägelse. Detta kan göra att modellen blir mindre känslig för variationer i data, vilket kan vara bra om det finns mycket brus. Men det kan också göra att modellen missar viktiga detaljer.

Till exempel, om du har K lika med storleken på ditt dataset, kommer din KNN-modell att förutsäga samma etikett för alla nya datapunkter, oavsett deras egenskaper. Etiketten kommer att vara den mest frekventa etiketten i ditt dataset.



##### #Overfitting 
Överanpassning innebär att modellen är för komplex och fångar upp brus och detaljer i träningsdata som inte är relevanta för det underliggande mönstret. Som ett resultat presterar en överanpassad modell bra på träningsdata men dåligt på ny, okänd data.

Till exempel, om du har K=1, kommer din KNN-modell att förutsäga etiketten för en ny datapunkt baserat på den närmaste datapunkten i träningsdata. Om denna närmaste datapunkt råkar vara ett outlier eller brus, kommer din modell att göra en felaktig förutsägelse.



##### #Training-set 
 Är en uppsättning data som används för att träna en maskininlärningsmodell. Till exempel med irisblommor, kan varje datapunkt representera en blomma och ha fyra funktioner: sepal längd, sepal bredd, petal längd och petal bredd. Då skulle en datapunkt kunna se ut så här: `[5.1, 3.5, 1.4, 0.2]`. Varje datapunkt i ditt träningsset skulle vara en sådan vektor, tillsammans med en etikett som anger vilken art blomman tillhör (t.ex. setosa, versicolor eller virginica).


##### #Test-set
Är en uppsättning data som används för att utvärdera prestandan hos en maskininlärningsmodell efter att den har tränats på ett träningsset. Till exempel irisblommor, skulle testsetet bestå av irisblommor vars arter är okända, och vi skulle använda våran tränade modell för att förutsäga arten för varje blomma i testsetet.
##### #Regression 
Handlar om att förutsäga numeriska resultat baserat på tidigare data och skapa en anpassad matematisk funktion för detta ändamål. Används för att lösa problem som att förutsäga priser, temperaturer eller andra kontinuerliga värden.


##### #Classification
Målet är att skapa en modell som kan förutsäga vilken kategori eller klass en given ingångsdata tillhör, baserat på tidigare träningsdata. Klassifikation används för att lösa problem som att  identifiera olika typer av växtarter baserat på bilder eller avgöra om en kund kommer att köpa en produkt eller inte. Det handlar om att sätta saker i rätt klass istället för att förutspå siffror som regression gör, klassifikation är också en övervakad inlärning.


##### #KNN 
Är en instansbaserad inlärningsalgoritm som används i ML. Den lagrar alla tillgängliga fall och klassificerar nya fall baserat på likamått med hjälp av distansen mellan träningsfallen. KNN har funnits sen 1970. Klassifikation och Regression är två olika tillämpningar på KNN, de används för olika problem.  
**Klassifikation** är övervakat lärande där vi vill hitta en etikett tex vad det är för djur eller för blomma.
**Regression** är också övervakat, men man vill hitta ett kontinuerligt värde. Istället för att räkna antal etiketter i de närmsta grannarna så vill du ta de närmsta grannarna värde och ta medelvärdet av det.
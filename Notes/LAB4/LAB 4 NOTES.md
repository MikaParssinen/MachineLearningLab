Linjär regression är en metod för att modellera relationen mellan en eller flera oberoende variabel och en beroende variabel genom att anpassa en linje till datan. I vårat fall har vi hus priser i boston som är våran data.  
I denna data hittar vi olika egenskaper/attribut, vilket är våra oberoende variabler eftersom dessa påverkar priset på huset. Exemplevis har vi storlek på huset, ålder på huset osv...

Våran uppgift är då att modellera priset på huset (MEDV) baserat på dessa oberoende variabler.




**What differences have you noticed between Matrix Form and Gradient descent methods? Which executes faster? Which approach gives better predictions?**

I matrix så använder vi matris multiplikation. Vi behöver inte itterera något utan endast en beräkning. Snabbare, ofta för små och medium stora datasets.

I Gradient descent tar vi små och många steg för att hitta vikten. Vi behöver itterera över all data många gånger. Används för stora datasets matris blir ineffektivft.

**Vilka värden blir bäst?:**

Bästa värden får vi 5000 epoker, 0,05

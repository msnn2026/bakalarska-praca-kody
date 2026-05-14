# Zdrojové kódy k bakalárskej práci

Tento repozitár obsahuje zdrojové kódy a experimenty použité v praktickej
časti bakalárskej práce zameranej na matematiku neurónových sietí.

## Obsah repozitára

- aproximačná neurónová sieť,
- autoregresívny znakový model,
- vplyv inicializácie váh na disperziu výstupu lineárnej vrstvy

## Poznámky k pôvodu implementácií

Myšlienky a časti kódu v tomto repozitári vychádzajú
zo vzdelávacieho kurzu Andreja Karpathyho *Neural Networks: Zero to Hero*.

Oficiálny repozitár:

https://github.com/karpathy/nn-zero-to-hero

Pôvodné materiály sú dostupné pod licenciou MIT. Licenčné oznámenie pôvodného
autora je uvedené v súbore [LICENSE-KARPATHY.md](LICENSE-KARPATHY.md).

Kódy v tomto repozitári boli upravené a spracované pre potreby bakalárskej
práce so zameraním na matematické vysvetlenie doprednej propagácie, spätnej
propagácie, optimalizácie a stability učenia neurónových sietí.

## Poznámky k použitému datasetu

Pri trénovaní autoregresívneho znakového modelu bol použitý dataset
`top-10000-websites.txt`, dostupný v repozitári projektu Protego:

https://github.com/scrapy/protego/blob/master/tests/top-10000-websites.txt

Dataset slúži iba ako vstupný tréningový súbor pre experimenty v praktickej
časti bakalárskej práce. V tomto repozitári nie je dataset upravovaný ani
redistribuovaný; vyššie uvádzame odkaz na pôvodný zdroj.

## Moje poznámky

- Snažil som sa jednotlivé kódy spracovať čo do najjednoduchšej podoby, aby im porozumeli aj študenti matematiky, ktorí s pythonom začínajú.
- Používam metódu "from scratch", resp. kódy nie sú optimalizované, ale za to sa im dá lepšie chápať
- V priebehu komentujem postupy, ktoré sú kľúčové (ľudskou rečou)
- Na niektoré manuálne časti kódov (grafy, vypisovanie údajov, opravy mojich výmyslov) som použil asistenciu UI (model GPT-5.5 Thinking)
- Problém s UI je, že dá extrémne optimalizovaný kód na polovicu riadkov (často tomu ako matematik nechápem)
- Najviac odporúčam online prednášky na YouTube (cca 20 hodín), potom oficiány GitHub s kódmi (tu sú aj modely optimalizované cez PyTorch) 
- Spomínam aj licencie, aj keď je moja práca hlavne edukatívna (nie publikačná)




Licencia pôvodného projektu Protego je dostupná v jeho oficiálnom repozitári:

https://github.com/scrapy/protego/blob/master/LICENSE

# Zdrojové kódy k bakalárskej práci

Tento repozitár obsahuje zdrojové kódy a experimenty použité v praktickej
časti bakalárskej práce zameranej na matematiku neurónových sietí.

## Obsah repozitára

- aproximačná neurónová sieť,
- autoregresívny znakový model,
- inicializácia parametrov a stabilita aktivácií,

## Poznámka k pôvodu implementácií

Implementačné myšlienky a časti kódu v tomto repozitári vychádzajú
zo vzdelávacieho kurzu Andreja Karpathyho *Neural Networks: Zero to Hero*.

Oficiálny repozitár:

https://github.com/karpathy/nn-zero-to-hero

Pôvodné materiály sú dostupné pod licenciou MIT. Licenčné oznámenie pôvodného
autora je uvedené v súbore [LICENSE-KARPATHY.md](LICENSE-KARPATHY.md).

Kódy v tomto repozitári boli upravené a spracované pre potreby bakalárskej
práce so zameraním na matematické vysvetlenie doprednej propagácie, spätnej
propagácie, optimalizácie a stability učenia neurónových sietí.

## Poznámka k použitému datasetu

Pri trénovaní autoregresívneho znakového modelu bol použitý dataset
`top-10000-websites.txt`, dostupný v repozitári projektu Protego:

https://github.com/scrapy/protego/blob/master/tests/top-10000-websites.txt

Dataset slúži iba ako vstupný tréningový súbor pre experimenty v praktickej
časti bakalárskej práce. V tomto repozitári nie je dataset upravovaný ani
redistribuovaný; vyššie uvádzame odkaz na pôvodný zdroj.

Licencia pôvodného projektu Protego je dostupná v jeho oficiálnom repozitári:

https://github.com/scrapy/protego/blob/master/LICENSE

# Clickbait Detection

Using scikit learn RandomForestClassifier to detect clickbait.

Data taken from here:
http://www.clickbait-challenge.org/

Sample output:
```
cross val:
[ 0.77972466  0.79197995  0.79673777]
roc auc:
0.844501860273
confusion:
[[454 136]
 [ 33 377]]
report:
              precision    recall  f1-score   support

   clickbait       0.93      0.77      0.84       590
no-clickbait       0.73      0.92      0.82       410

 avg / total       0.85      0.83      0.83      1000

false positive target titles:
LeBron James' personal barber: 'There is no dye'
Natasha Lyonne on Meeting Nicki Minaj, Preparing for “Orange Is The New Black” & Why She Hates Spanx  [Video]
Jaden Smith joins Baz Luhrmann's Netflix drama
‘Jurassic World’ Bites Off Biggest Global Debut of All Time With $511.8 Million
All the news from Apple's WWDC 2015 Keynote
Homer and Marge aren't over on 'The Simpsons' because love conquers all
‘The Art of Dissent’ 
Throwing a Lifeline for the High Seas
Bryce Dallas Howard Cries On Command While Talking Home Depot With Conan
Lost Brother in Yosemite 
Cleveland Weatherman Voices Displeasure Over LeBron James' No-Call Foul on TV
Magic Johnson says LeBron might be having greatest Finals ever, and Curry and Green need to keep quiet - SportsNation - ESPN
Jurassic World returns for new generation
An Open Letter to Jerry Seinfeld from a 'Politically Correct' College Student
Another Bonus of Best Friends: They're Good for Your Health
Android Auto Review: Your Next Car Needs This [Video]
Vine
Armpit hair, don't care! Chinese women flood social media with hairy underarm selfies to prove they don't need to be hairless to be beautiful
10 Ways the Expat Life Is Like a Continual Espresso Buzz - Expat
Ranking the MLB teams ahead of the offseason
Blake Griffin Explains the Lyrics to Fetty Wap's 'Trap Queen'
Super PACs explained
19 Reasons Why Christopher Lee Was A Pretty Amazing Human Being
Apple Music's rivals aren't impressed
Natasha Lyonne on Meeting Nicki Minaj, Preparing for “Orange Is The New Black” & Why She Hates Spanx  [Video]
Defining moments in '70s television
Secret Cinema: The Empire Strikes Back review - the force is weak with this one
Light on your feet! Japanese inventor creates LED dancing shoes that allow users to paint the town any colour they like
UGA's Nick Chubb Doesn't Use Social Media Because He Doesn't Need to Advertise
Simple Life Among the Hutterites 
The Koch brothers and the Republican Party go to war — with each other
How the Spurs' 2014 Finals performance changed the NBA forever
Scouting Reports for Riley Curry and the All-NBA Kid Team
```

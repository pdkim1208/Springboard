**Capstone 1 Milestone Report**

***1. Project Overview***

The purpose of this project exercise is to find trends between
performance at the NFL Combine and how highly these players are drafted.
In doing so, both the teams and prospective players both have a clearer
understanding of what this historical data looks like and how they stand
comparatively to past performances. For players, these insights will
serve to give clear benchmarks to help shape their training goals, as
well as give proper perspective on how they stand relative to past
performances. For NFL teams, this project will give a more in-depth,
multi-dimensional, and quantitative methodology for understanding how
great or poor an NFL Combine performance is.

Every major evaluator of talent (scout, General Manager, head coach,
player personnel staff, etc) usually has a strong ability to make a
fairly consistent heuristic analysis of what level a player is based on
their Combine measurables. For example, any strength and conditioning
coach at the college level could probably break down what are solid
combine numbers for a player at a given position to get drafted.
However, the ability for any human brain to process and compute the
quantitative relationships of performance and Draft selection across
many important dimensions like weight, height, and position becomes
incredibly difficult at a large scale.

This project will be broken up into two major phases – the first phase
is the Exploratory Data Analysis (EDA) phase, where we will explore
trends and find patterns in the data. At this phase, we will quantify
and reveal the performance benchmarks so that both stakeholders have an
empirical reference point for judging performance, rather than a
heuristic one. In the second phase of this project, we will apply
machine learning methodologies to take in a test dataset and try to
classify them by which Round group (selected in Rounds 1-3, 4-7, or
Undrafted) they will be selected in using a multiclass classifier.

***2. Introduction and Background***

The NFL Combine is the premier event for prospective college football
athletes to showcase their athletic abilities to their future NFL team.
This annual event attracts scouts, coaches, team owners, and team
executives to Indianapolis, all in hopes of finding and evaluating the
future players that will carry the future of their franchises. In
addition, the NFL Combine is televised to give the general public a
first-hand look into the raw athletic abilities of some of the best
athletes on the planet.

Each year, the NFL's Selection Committee thoroughly examines thousands
of hours of college football film to identify the players with the
highest potential to be drafted, then sends out a combine invitation to
those selected players. With rare exceptions, the only situations in
which players decline to show up for the NFL Combine are debilitating
injuries/illnesses or family emergencies.

The NFL Combine offers players the option to choose which exercises to
participate in. There is a standard core set of measurable exercises,
which is the main focus of this project, as well as subjective drills
(ie throwing passes, interception drills, etc). The measured exercises
that we will focus on are the following:

1.  40 Yard Dash - Time, in seconds, that a player covers 40 yards

2.  Bench Press - The number of repetitions that a player is able to
    bench press 225 pounds until failure

3.  Vertical Leap - The number of inches that a player is able to jump
    from a standing position

4.  Broad Jump - The number of inches that a player jumps forward and
    land squarely on both feet

5.  Shuttle - The time, in seconds, it takes a player to run 5 yards,
    then 10 yards in the opposite direction, then 5 yards back. This
    exercise is used to measure lateral quickness.

6.  3Cone Drill - The time, in seconds, it takes a player to make an L
    shape through 3 cones.

After the NFL Combine, each team has an extensive review process where
they will convene with the necessary personnel and evaluate a draft
strategy based on their scouting reports. Furthermore, teams will invite
prospects for private workouts and interviews to evaluate intangible
characteristics like leadership and character. After about a month from
the combine, the NFL hold its annual draft, which consists of 7 rounds
of about 32 players each (\~220 players drafted) in which teams go in a
predetermined order and select players in each round, where the teams
with the worst regular season records usually go first in selecting
available players in the selection pool. The difference in contract
values in each of these rounds is immense. Below is a table of contracts
for the first pick of each round:

  ***Name***       ***Round***   ***Contract Length (Years)***   ***Total Contract Value (mm)***
  ---------------- ------------- ------------------------------- ---------------------------------
  Myles Garrett    1             4                               \$30
  Kevin King       2             4                               \$7
  Larry Ogunjobi   3             4                               \$3.9
  Vince Biegel     4             4                               \$3.1
  Jake Butt        5             4                               \$2.7
  Caleb Brantley   6             4                               \$2.6
  Coley Stacy      7             4                               \$2.5

From a player’s perspective, the importance of the round drafted and
knowing where the player stands relative to competitors is fully
captured in the table above. Equally important is the ownership’s
perspective, as teams’ successes can often be carried on the backs of a
few superstar players. Conversely, every owner’s worst nightmare is
drafting a bust. A bust is defined as a player who shows incredible
promise, is touted as the next superstar, is given an outsized contract,
then fails to deliver production anywhere close to expectations. Not
only are these players a financial drain on an organization, as most
early round contracts require large guarantees, but also, the
opportunity cost is equally high as well. Since every bad player picked
means that the competition is choosing from a pool of more productive
players, this is often a nightmare situation for low-performing
franchises (ie Cleveland Browns). Every NFL team is worth at least a
billion dollars and the revenue implications between a playoff team and
a non-playoff team lies in the hundreds of millions of dollars. This
project is attempting to address the pressing need for both stakeholders
(players and owners) to understand the detailed statistics behind the
NFL Combine and Draft.

***3. Initial Hypotheses***

There are a number of hypotheses, ranging from self-obvious to ones that
delve into the deep domain knowledge of football, that we are going to
explore and test in this project.

1.  The first, most self-obvious hypothesis that we will test is whether
    performance at the combine translates into a higher draft status.
    Even though this may seem self-obvious, there are actually quite a
    few caveats to this statement. For example, every position has a
    different degree of dependency on pure athleticism or specific
    combine drills as accurate predictors of future success. Defensive
    backs are scrutinized on Combine metrics such as 40-yard dash and
    vertical leap, as their primary functions are to contain speedy wide
    receivers and leap high to defend against passes. These metrics are
    considered bonuses for quarterbacks, as a dual-threat quarterback
    presents an extra element to an offense, but are no means considered
    a requirement like they are for defensive backs. We can quantify
    these distinctions by analyzing the regression lines of players in
    each draft group (Rounds 1-3, 4-7, and Undrafted).

2.  The second hypothesis is fairly self-obvious, but necessary for
    building linear models. As weight is one of the most important
    factors for giving context to a Combine drill performance, we can
    plot each drill against weight to see if a correlation exists. One
    could expect positive correlations in drills such as 40-yard dash,
    bench press, 3Cone, and shuttle, and negative correlations in
    vertical leap and broad jump.

3.  The third hypothesis delves deeper into the nuances of football. It
    is that positions most dependent on measurables and less on niche
    skills (Offensive Linemen and Defensive Linemen) will have less
    variance and spread, and will have a higher R-squared coefficient.
    Conversely, skill positions where one can overcome physical
    deficiencies with ball skills, should exhibit a larger spread, and
    in turn, a lower R-squared coefficient.

***4. Data Wrangling***

Two different datasets, NFL\_Draft\_Rounds.csv and
CombineMeasurables.csv, are being used for analysis.
CombineMeasurables.csv contains all the data around the performance of
each player at the draft, while NFL\_Draft\_Rounds.csv contains the
Round that each player has been drafted in. CombineMeasuarable.csv
contains data from 1987 to 2017, while NFL\_Draft\_Rounds contains data
from 1985.

CombineMeasurables contains the following columns:

Year, Name, College, POS, Height, Weight, Wonderlic, 40\_Yard,
Bench\_Press, Vert\_Leap, Broad\_Jump, Shuttle, 3Cone

NFL\_Draft\_Rounds contains the following columns:

Player\_Id, Year, Rnd, Pick, Tm, Player, Pos, Position Standard,
First4AV, Age, To, AP1, PB, St, CarAV, DrAV, G, Cmp, Pass\_Att,
Pass\_Yds, Pass\_Int, Rush\_Att, Rush\_Yds, Rush\_TDs, Rec, Rec\_Yds,
Rec\_Tds, Tkl, Def\_Int, Sk, College/Univ.

The first major item of cleanup was dropping the columns that were
outside the scope of the project. Since the purpose of the project was
to build a prediction model around combine performance and not college
statistics, we will drop all columns containing game statistics. We kep
the following columns from combine\_measurables – 'Year', 'Name', 'POS',
'Height', 'Weight', 'Wonderlic', '40\_Yard', 'Bench\_Press',
'Vert\_Leap', 'Broad\_Jump', 'Shuttle', '3Cone'. From the rounds
dataframe, we selected the following columns - 'Player', 'Year', 'Rnd',
'Pos'.

To make the join operation easier (we will be doing a left join in order
to attach the round in which the player was selected to the combine the
the round dataframe), we renamed columns in the combine\_measurables
dataframe to match that of the rounds dataframe as such:

combine\_measurables = combine\_measurables.rename(index = str, columns
= {'Name': 'Player', 'POS':'Pos'})

Next, select only entries from the rounds dataframe that contains values
from any year before 1987, which is the first year that official NFL
Combine data is available. In addition, only select observations where
the value in ‘Rnd’ of the rounds dataframe is less than 7, since the NFL
Draft used be 12 rounds instead of today’s 7. As we want to standardize
everything to today’s version, we will only select observations where
the player is selected in rounds 1-7.

A very important component of wrangling the datasets involves
understanding the different position. We can use nuniques() and
uniques() on the “Position” column to obtain these values. Because we
will be performing a join operation on these datasets with position as
one of the “on” columns, this step is very crucial. What I did was
consolidate position groups into a superset, using standard supersets of
position groups with similar functions, weight, height, and measurable.

C, G, LS, T, OL, OT OL

DE, DL, DT, NT DL

CB, DB, SS, FS DB

The method below can be used on the ‘Pos’ column of both dataframes to
convert each position group to its proper superset

def consolidate(i):

if i == 'CB' or i == 'FS' or i == 'SS':

return 'DB'

elif i == 'ILB' or i == 'OLB':

return 'LB'

elif i == 'NT' or i == 'DE' or i == 'DT':

return 'DL'

elif i == 'C' or i == 'LS' or i == 'G' or i == 'T' or i == 'OT':

return 'OL'

else:

return str(i)

A minor consideration was the elimination of punters (P) and kickers
(K), as they both have a very small sample size, largely null values,
and a very low correlation between NFL combine measurable with NFL
success (as they are niche skill positions).

Once both dataframes are ready, we will execute the left join to append
the rounds in which each player was drafted to the combine data. We need
to keep the NaN values (which is why we use the left join), as we can
later change these to Undrafted.

df\_joined = combine\_measurables.merge(rounds, how = 'left', on =
\['Year', 'Player', 'Pos'\])

We drop the columns ‘Year’ and ‘Player’, which previously served as
unique identifiers, as we would not be making any analysis by those
dimensions.

df\_joined = df\_joined\[\['Rnd', 'Pos', 'Height', 'Weight',
'Wonderlic', '40\_Yard', 'Bench\_Press', 'Vert\_Leap', 'Broad\_Jump',
'Shuttle', '3Cone'\]\]

In order to reduce the number of dimensions to make the data more
comprehensible, we consolidate the rounds that the players were drafted
in as a string in the following groups (1-3, 4-7, Undrafted):

def consolidate\_rounds(i):

if i == 1 or i == 2 or i == 3:

return '1-3'

elif i == 4 or i == 5 or i == 6 or i == 7:

return '4-7'

else:

return 'Undrafted'

An important aspect of the NFL Combine to note is that prospects don’t
have to complete all of the drills. For example, Deion Sanders, a
high-profile first-round draft pick and first ballot Hall of Famer,
famously arrived in a limo, ran the 40-yard dash, saluted the scouts,
then got back into his limo and back to his private plane. Thus, he only
had one measurable, with all the rest as NA. We deal with this
unavailable data using dropna(), but we do this once we do a more
in-depth analysis.

There weren’t very many outliers, as the NFL has a selection committee
to determine which prospects are likely to be drafted and have a serious
chance to showcase their skills. Also, since the data in question
pertains to the physical capabilities of the most elite athletes on the
planet, the spread of the data fits in a relatively predictable range.
Therefore, having outliers skew the data is not necessarily a major
concern of this exercise. There were two minor outliers, however, in the
Broad Jump exercise in which two values of 22 and 30 inches were
recorded, which could have either been a mistaken entry or an
unrepresentative performance, as even an average 5^th^ grade student can
broad jump more than 36 inches.

***5. Exploratory Data Analysis***

The data analysis and visualization from this project aims to provide
in-depth, empirical analysis on the relationship between performance at
the NFL Combine and performance in the NFL Draft. We will do so by
grouping players by their positions, then creating violinplots (to show
the range and concentration of each position group) and running
regressions on the scatterplots across various exercises (to show the
spread of the data and arrive at a function for the best fit line).

The following heights and weights for each position group are found in
the violinplots below:

![](media/image1.tiff){width="3.3777777777777778in"
height="2.3878915135608048in"}![](media/image2.tiff){width="3.2876334208223974in"
height="2.3422255030621173in"}

The following are violinplots (ordered by highest performing position
group) and corresponding scatterplots (with performance against weight):

***40-Yard Dash***

![](media/image3.tiff){width="3.468687664041995in"
height="2.484086832895888in"}
![](media/image4.tiff){width="3.432169728783902in"
height="2.4263418635170604in"}

**Regression for Undrafted:** y=0.0058x + 3.484, R-Squared = 0.688

**Regression for Rounds 4-7:** y=0.0059x + 3.387, R-Squared = 0.763

**Regression for Rounds 1-3:** y=0.0056x + 3.366, R-Squared = 0.759

***Bench Press***

![](media/image5.tiff){width="3.68125in"
height="2.657069116360455in"}![](media/image6.tiff){width="3.6652930883639545in"
height="2.6455522747156603in"}

**Regression for Undrafted**: y=0.0899x + -3.105, R-Squared = 0.368

**Regression for Rounds 4-7**: y=0.08958x + -2.074, R-Squared = 0.392

**Regression for Rounds 1-3**: y=0.0919x + -1.857, R-Squared = 0.427

***Vertical Leap***

![](media/image7.tiff){width="3.70205927384077in"
height="2.672088801399825in"}
![](media/image8.tiff){width="3.6865693350831146in"
height="2.6609109798775155in"}

**Regression for Undrafted**: y=-0.0536x + 44.0126, R-Squared = 0.322

**Regression for Rounds 4-7**: y=-0.0565x + 45.885, R-Squared = 0.387

**Regression for Rounds 1-3**: y=-0.0583x + 47.449, R-Squared = 0.4137

***Broad Jump***

![](media/image9.tiff){width="3.68125in"
height="2.622648731408574in"}![](media/image10.tiff){width="3.6564271653543305in"
height="2.60496719160105in"}

**Regression for Undrafted**: y=-0.132x + 142.188, R-Squared = 0.407

**Regression for Rounds 4-7**: y=-0.139x + 145.950, R-Squared = 0.481

**Regression for Rounds 1-3**: y=-0.139x + 148.872, R-Squared = 0.485

***Shuttle***

![](media/image11.tiff){width="3.6990430883639545in"
height="2.649054024496938in"}
![](media/image12.tiff){width="3.6865693350831146in"
height="2.6061876640419945in"}

**Regression for Undrafted**: y=0.00449x + 3.361, R-Squared = 0.526

**Regression for Rounds 4-7**: y=0.00434x + 3.338, R-Squared = 0.561

**Regression for Rounds 1-3**: y=0.00435x + 3.289, R-Squared = 0.594

***3Cone***

![](media/image13.tiff){width="3.68125in" height="2.636312335958005in"}
![](media/image14.tiff){width="3.6901148293963253in"
height="2.642660761154856in"}

**Regression for Undrafted**: y=0.00764x + 5.543, R-Squared = 0.531

**Regression for Rounds 4-7**: y=0.00714x + 5.569, R-Squared = 0.572

**Regression for Rounds 1-3**: y=0.00710x + 5.498, R-Squared = 0.613

Furthermore, it equally important to dive deeper and see how this data
plays out at the position group level.

***Defensive Backs***

  ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------
  ![](media/image15.tiff){width="3.4298392388451444in" height="2.456266404199475in"}    ![](media/image16.tiff){width="3.261075021872266in" height="2.4068471128608926in"}
                                                                                        
  Regression (Undrafted): y=0.001805x + 4.269, R^2^ = 0.0432                            Regression (Undrafted): y=0.129x + -11.810, R^2^ = 0.137
                                                                                        
  Regression (Rounds 4-7): y=0.000916x + 4.376, R^2^ = 0.0160                           Regression (Rounds 4-7): y=0.173x + -19.442, R^2^ = 0.221
                                                                                        
  Regression (Rounds 1-3): y=0.00112x + 4.277, R^2^ = 0.0229                            Regression (Rounds 1-3): y=0.139x + -12.465, R^2^= 0.120
  ------------------------------------------------------------------------------------- -------------------------------------------------------------------------------------
  ![](media/image17.tiff){width="3.516521216097988in" height="2.4859711286089237in"}    ![](media/image18.tiff){width="3.3556681977252842in" height="2.390695538057743in"}
                                                                                        
  Regression (Undrafted): y=0.00311x + 33.519, R^2^ = 0.000198                          Regression (Undrafted): y=0.0202x + 113.229, R^2^ = 0.00238
                                                                                        
  Regression (Rounds 4-7): y=0.01575x + 32.110, R^2^ = 0.00504                          Regression (Rounds 4-7): y=0.0232x + 114.674, R^2^ = 0.00264
                                                                                        
  Regression (Rounds 1-3): y=0.0134x + 33.581, R^2^ = 0.00295                           Regression (Rounds 1-3): y=0.0190x + 118.075, R^2^ = 0.00143

  ![](media/image19.tiff){width="3.5353794838145234in" height="2.5318471128608926in"}   ![](media/image20.tiff){width="3.2261439195100614in" height="2.3103893263342083in"}
                                                                                        
  Regression (Undrafted): y=0.000526x + 4.134, R^2^ = 0.001988                          Regression (Undrafted): y=0.000290x + 7.0417, R^2^ = 0.000205
                                                                                        
  Regression (Rounds 4-7): y=0.000298x + 4.133, R^2^ = 0.000687                         Regression (Rounds 4-7): y=0.00215x + 6.598, R^2^ = 0.0131
                                                                                        
  Regression (Rounds 1-3): y=0.000335x + 4.068, R^2^ = 0.000607                         Regression for Rounds 1-3: y=0.00121x + 6.693, R^2^ = 0.00321
  ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------

***Offensive Lineman***

  -------------------------------------------------------------------------------------------------------------------------------------------------------------------------
  ![](media/image21.tiff){width="3.397818241469816in" height="2.433333333333333in"}   ![](media/image22.tiff){width="3.4251990376202976in" height="2.4722583114610672in"}
                                                                                      
  Regression (Undrafted): y=0.00381x + 4.192, R^2^ = 0.155                            Regression (Undrafted): y=0.0638x + 3.686, R^2^ = 0.0577
                                                                                      
  Regression (Rounds 4-7): y=0.00375x + 4.131, R^2^ = 0.157                           Regression (Rounds 4-7): y=0.0415x + 11.794, R^2^ = 0.0175
                                                                                      
  Regression (Rounds 1-3): y=0.00375x + 4.059, R^2^ = 0.153                           Regression (Rounds 1-3): y=0.0368x + 13.806, R^2^ = 0.0181
  ----------------------------------------------------------------------------------- -------------------------------------------------------------------------------------
  ![](media/image23.tiff){width="3.57958552055993in" height="2.5305555555555554in"}   ![](media/image24.tiff){width="3.351213910761155in" height="2.387522965879265in"}
                                                                                      
  Regression (Undrafted): y=-0.0198x + 32.463, R^2^ = 0.0185                          Regression (Undrafted): y=-0.1129x + 133.530, R^2^ = 0.108
                                                                                      
  Regression (Rounds 4-7): y=-0.0185x + 33.246, R^2^ = 0.0122                         Regression (Rounds 4-7): y=-0.102x + 132.345, R^2^ = 0.0780
                                                                                      
  Regression (Rounds 1-3): y=-0.0253x + 36.201, R^2^ = 0.0252                         Regression (Rounds 1-3): y=-0.0652x + 123.455, R^2^ = 0.0351

  ![](media/image25.tiff){width="3.51409886264217in" height="2.4842596237970254in"}   ![](media/image26.tiff){width="3.4689293525809273in" height="2.4842596237970254in"}
                                                                                      
  Regression (Undrafted): y=0.00370x + 3.674, R^2^ = 0.119                            Regression (Undrafted): y=0.00677x + 5.940, R^2^ = 0.073
                                                                                      
  Regression (Rounds 4-7): y=0.00470x + 3.285, R^2^ = 0.140                           Regression (Rounds 4-7): y=0.0102x + 4.690, R^2^ = 0.170
                                                                                      
  Regression (Rounds 1-3): y=0.00455x + 3.270, R^2^ = 0.175                           Regression (Rounds 1-3): y=0.0104x + 4.5448, R^2^ = 0.213
  -------------------------------------------------------------------------------------------------------------------------------------------------------------------------

***Defensive Linemen***

  ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------
  ![](media/image27.tiff){width="3.417436570428696in" height="2.4284109798775155in"}    ![](media/image28.tiff){width="3.3318788276465443in" height="2.386111111111111in"}
                                                                                        
  Regression (Undrafted): y=0.00535x + 3.546, R^2^ = 0.373                              Regression (Undrafted): y=0.0756x + 1.879, R^2^ = 0.102
                                                                                        
  Regression (Rounds 4-7): y=0.00560x + 3.423, R^2^ = 0.440                             Regression (Rounds 4-7): y=0.0877x + -1.067, R^2^ = 0.117
                                                                                        
  Regression (Rounds 1-3): y=0.00553x + 3.362, R^2^ = 0.440                             Regression (Rounds 1-3): y=0.111x + -6.427, R^2^ = 0.188
  ------------------------------------------------------------------------------------- -------------------------------------------------------------------------------------
  ![](media/image29.tiff){width="3.357176290463692in" height="2.4042279090113734in"}    ![](media/image30.tiff){width="3.468254593175853in" height="2.4518514873140855in"}
                                                                                        
  Regression (Undrafted): y=-0.0693x + 49.275, R^2^ = 0.202                             Regression (Undrafted): y=-0.177x + 157.280, R^2^ = 0.283
                                                                                        
  Regression (Rounds 4-7): y=-0.0728x + 51.214, R^2^ = 0.215                            Regression (Rounds 4-7): y=-0.1964x + 164.096, R^2^ = 0.350
                                                                                        
  Regression (Rounds 1-3): y=-0.0635x + 49.278, R^2^ = 0.190                            Regression (Rounds 1-3): y=-0.204x + 168.655, R^2^ = 0.360

  ![](media/image31.tiff){width="3.5220909886264216in" height="2.5027777777777778in"}   ![](media/image32.tiff){width="3.5220909886264216in" height="2.5027777777777778in"}
                                                                                        
  Regression (Undrafted): y=0.00515x + 3.129, R^2^ = 0.280                              Regression (Undrafted): y=0.00984x + 4.845, R^2^ = 0.333
                                                                                        
  Regression (Rounds 4-7): y=0.00409x + 3.380, R^2^ = 0.203                             Regression (Rounds 4-7): y=0.00825x + 5.219, R^2^ = 0.268
                                                                                        
  Regression (Rounds 1-3): y=0.00457x + 3.210, R^2^ = 0.276                             Regression (Rounds 1-3): y=0.00933x + 4.808, R^2^ = 0.372
  ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------

***Tight Ends***

  ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------
  ![](media/image33.tiff){width="3.523659230096238in" height="2.5234536307961504in"}    ![](media/image34.tiff){width="3.6005818022747156in" height="2.5988451443569556in"}
                                                                                        
  Regression (Undrafted): y=0.00407x + 3.875, R^2^ = 0.117                              Regression (Undrafted): y=0.0919x + -5.328, R^2^ = 0.0549
                                                                                        
  Regression (Rounds 4-7): y=0.00472x + 3.631, R^2^ = 0.127                             Regression (Rounds 4-7): y=0.111x + -8.313, R^2^ = 0.0662
                                                                                        
  Regression (Rounds 1-3): y=0.00390x + 3.745, R^2^ = 0.0700                            Regression (Rounds 1-3): y=0.0633x + 5.495, R^2^ = 0.0152
  ------------------------------------------------------------------------------------- -------------------------------------------------------------------------------------
  ![](media/image35.tiff){width="3.5635061242344706in" height="2.5720844269466316in"}   ![](media/image36.tiff){width="3.6808398950131234in" height="2.6223600174978126in"}
                                                                                        
  Regression (Undrafted): y=-0.0471x + 42.636, R^2^ = 0.0417                            Regression (Undrafted): y=-0.1261x + 141.671, R^2^ = 0.0735
                                                                                        
  Regression (Rounds 4-7): y=-0.0397x + 42.166, R^2^ = 0.0224                           Regression (Rounds 4-7): y=-0.0870x + 134.887, R^2^ = 0.0320
                                                                                        
  Regression (Rounds 1-3): y=-0.0278x + 40.766, R^2^ = 0.00558                          Regression (Rounds 1-3): y=-0.0757x + 135.556, R^2^ = 0.0111

  ![](media/image37.tiff){width="3.669332895888014in" height="2.6277777777777778in"}    ![](media/image38.tiff){width="3.717110673665792in" height="2.6277777777777778in"}
                                                                                        
  Regression (Undrafted): y=0.00392x + 3.464, R^2^ = 0.0951                             Regression for Undrafted: y=0.00426x + 6.279, R^2^ = 0.0297
                                                                                        
  Regression (Rounds 4-7): y=0.00348x + 3.490, R^2^ = 0.0564                            Regression for Rounds 4-7: y=0.00823x + 5.127, R^2^ = 0.0827
                                                                                        
  Regression (Rounds 1-3): y=0.00302x + 3.553, R^2^ = 0.0210                            Regression for Rounds 1-3: y=0.00484x + 5.907, R^2^ = 0.0335
  ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------

***Running Backs***

  ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------
  ![](media/image39.tiff){width="3.6305457130358705in" height="2.6in"}                  ![](media/image40.tiff){width="3.594484908136483in" height="2.5944444444444446in"}
                                                                                        
  Regression (Undrafted): y=0.00260x + 4.086, R^2^ = 0.109                              Regression (Undrafted): y=0.127x + -9.720, R^2^ = 0.149
                                                                                        
  Regression (Rounds 4-7): y=0.00217x + 4.116, R^2^ = 0.0922                            Regression (Rounds 4-7): y=0.0988x + -2.640, R^2^ = 0.106
                                                                                        
  Regression (Rounds 1-3): y=0.00190x + 4.114, R^2^ = 0.0703                            Regression (Rounds 1-3): y=0.0939x + -0.771, R^2^ = 0.0817
  ------------------------------------------------------------------------------------- -------------------------------------------------------------------------------------
  ![](media/image41.tiff){width="3.607176290463692in" height="2.550061242344707in"}     ![](media/image42.tiff){width="3.498694225721785in" height="2.4925929571303587in"}
                                                                                        
  Regression (Undrafted): y=-0.0110x + 35.702, R^2^ = 0.00290                           Regression (Undrafted): y=-0.0496x + 125.777, R^2^ = 0.0141
                                                                                        
  Regression (Rounds 4-7): y=-0.000859x + 34.241, R^2^ = 0.0000175                      Regression (Rounds 4-7): y=-0.0340x + 123.885, R^2^ = 0.00777
                                                                                        
  Regression (Rounds 1-3): y=0.0187x + 30.945, R^2^ = 0.00721                           Regression (Rounds 1-3): y=0.0239x + 114.495, R^2^ = 0.00375

  ![](media/image43.tiff){width="3.4947878390201224in" height="2.5027777777777778in"}   ![](media/image44.tiff){width="3.5402941819772527in" height="2.5027777777777778in"}
                                                                                        
  Regression (Undrafted): y=0.00256x + 3.747, R^2^ = 0.0475                             Regression (Undrafted): y=0.00182x + 6.785, R^2^ = 0.0112
                                                                                        
  Regression (Rounds 4-7): y=0.00234x + 3.764, R^2^ = 0.0563                            Regression (Rounds 4-7): y=0.00607x + 5.810, R^2^ = 0.0989
                                                                                        
  Regression (Rounds 1-3): y=0.00208x + 3.785, R^2^ = 0.0369                            Regression (Rounds 1-3): y=0.00249x + 6.479, R^2^ = 0.0184
  ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------

***Wide Receivers***

  -------------------------------------------------------------------------------------------------------------------------------------------------------------------------
  ![](media/image45.tiff){width="3.7074015748031495in" height="2.6008552055993in"}     ![](media/image46.tiff){width="3.602181758530184in" height="2.6in"}
                                                                                       
  Regression (Undrafted): y=0.000375x + 4.538, R^2^ = 0.00349                          Regression (Undrafted): y=0.131x + -13.443, R^2^ = 0.207
                                                                                       
  Regression (Rounds 4-7): y=0.000906x + 4.347, R^2^ = 0.0219                          Regression (Rounds 4-7): y=0.148x + -15.917, R^2^ = 0.208
                                                                                       
  Regression (Rounds 1-3): y=0.000940x + 4.299, R^2^ = 0.0241                          Regression (Rounds 1-3): y=0.101x + -4.906, R^2^ = 0.0833
  ------------------------------------------------------------------------------------ ------------------------------------------------------------------------------------
  ![](media/image47.tiff){width="3.4771708223972in" height="2.4518514873140855in"}     ![](media/image48.tiff){width="3.521751968503937in" height="2.4518514873140855in"}
                                                                                       
  Regression (Undrafted): y=0.0299x + 27.454, R^2^ = 0.0282                            Regression (Undrafted): y=0.0647x + 104.042, R^2^ = 0.0346
                                                                                       
  Regression (Rounds 4-7): y=0.0185x + 31.238, R^2^ = 0.0101                           Regression (Rounds 4-7): y=0.0602x + 107.439, R^2^ = 0.0277
                                                                                       
  Regression (Rounds 1-3): y=0.0183x + 32.136, R^2^ = 0.00920                          Regression (Rounds 1-3): y=0.0819x + 104.827, R^2^ = 0.0519

  ![](media/image49.tiff){width="3.440509623797025in" height="2.4074814085739282in"}   ![](media/image50.tiff){width="3.4669466316710413in" height="2.450926290463692in"}
                                                                                       
  Regression (Undrafted): y=0.000816x + 4.0959, R^2^ = 0.00830                         Regression (Undrafted): y=-0.000200x + 7.133, R^2^ = 0.000119
                                                                                       
  Regression (Rounds 4-7): y=0.00115x + 3.978, R^2^ = 0.0162                           Regression (Rounds 4-7): y=0.00131x + 6.728, R^2^ = 0.00607
                                                                                       
  Regression (Rounds 1-3): y=0.00226x + 3.739, R^2^ = 0.0500                           Regression (Rounds 1-3): y=0.000239x + 6.903, R^2^ = 0.000266
  -------------------------------------------------------------------------------------------------------------------------------------------------------------------------

***Quarterbacks***

+-----------------------------------+-----------------------------------+
| ![](media/image51.tiff){width="3. | ![](media/image52.tiff){width="3. |
| 5238429571303587in"               | 496065179352581in"                |
| height="2.5235859580052495in"}\   | height="2.523406605424322in"}     |
| Regression (Undrafted):           |                                   |
| y=0.00200x + 4.486, R^2^ = 0.0248 | Regression (Undrafted): y=0.379x  |
|                                   | + -68.181, R^2^ = 0.678           |
| Regression (Rounds 4-7):          |                                   |
| y=0.000923x + 4.673, R^2^ =       | Regression (Rounds 4-7): y=0.532x |
| 0.00436                           | + -97.549, R^2^ = 0.712           |
|                                   |                                   |
| Regression (Rounds 1-3):          | Regression (Rounds 1-3): y=0.576x |
| y=0.00112x + 4.562, R^2^ =        | + -110.134, R^2^ = 0.816          |
| 0.00597                           |                                   |
+===================================+===================================+
| ![](media/image53.tiff){width="3. | ![](media/image54.tiff){width="3. |
| 4682567804024496in"               | 5145833333333334in"               |
| height="2.4518514873140855in"}    | height="2.503912948381452in"}     |
|                                   |                                   |
| Regression (Undrafted):           | Regression (Undrafted):           |
| y=0.000595x + 29.815, R^2^ =      | y=-0.0226x + 112.751, R^2^ =      |
| 0.00000594                        | 0.00204                           |
|                                   |                                   |
| Regression (Rounds 4-7):          | Regression (Rounds 4-7):          |
| y=-0.0108x + 32.680, R^2^ =       | y=-0.00787x + 109.635, R^2^ =     |
| 0.00136                           | 0.000185                          |
|                                   |                                   |
| Regression (Rounds 1-3):          | Regression (Rounds 1-3):          |
| y=0.0235x + 26.631, R^2^ =        | y=0.0716x + 95.420, R^2^ = 0.0199 |
| 0.00918                           |                                   |
+-----------------------------------+-----------------------------------+
| ![](media/image55.tiff){width="3. | ![](media/image56.tiff){width="3. |
| 4947878390201224in"               | 5402930883639545in"               |
| height="2.5027777777777778in"}    | height="2.5027777777777778in"}    |
|                                   |                                   |
| Regression (Undrafted):           | Regression (Undrafted):           |
| y=0.00214x + 3.921, R^2^ = 0.0256 | y=0.00333x + 6.524, R^2^ = 0.0141 |
|                                   |                                   |
| Regression (Rounds 4-7):          | Regression (Rounds 4-7):          |
| y=0.000922x + 4.143, R^2^ =       | y=-0.00197x + 7.661, R^2^ =       |
| 0.00386                           | 0.00605                           |
|                                   |                                   |
| Regression (Rounds 1-3):          | Regression (Rounds 1-3):          |
| y=0.000700x + 4.123, R^2^ =       | y=0.000916x + 6.910, R^2^ =       |
| 0.00261                           | 0.00157                           |
+-----------------------------------+-----------------------------------+

***Linebackers***

  -------------------------------------------------------------------------------------------------------------------------------------------------------------------------
  ![](media/image57.tiff){width="3.5723632983377076in" height="2.558333333333333in"}   ![](media/image58.tiff){width="3.544455380577428in" height="2.558333333333333in"}
                                                                                       
  Regression (Undrafted): y=0.00246x + 4.224, R^2^ = 0.0416                            Regression for Undrafted: y=0.121x + -8.666, R^2^ = 0.0787
                                                                                       
  Regression for Rounds 4-7: y=0.00276x + 4.0875, R^2^ = 0.0424                        Regression for Rounds 4-7: y=0.0996x + -2.236, R^2^ = 0.0306
                                                                                       
  Regression for Rounds 1-3: y=0.00174x + 4.267, R^2^ = 0.0209                         Regression for Rounds 1-3: y=0.0972x + -1.409, R^2^ = 0.0446
  ------------------------------------------------------------------------------------ ------------------------------------------------------------------------------------
  ![](media/image59.tiff){width="3.453373797025372in" height="2.4925929571303587in"}   ![](media/image60.tiff){width="3.49999343832021in" height="2.4935181539807525in"}
                                                                                       
  Regression for Undrafted: y=-0.00338x + 33.089, R^2^ = 0.000137                      Regression for Undrafted: y=-0.0528x + 125.831, R^2^ = 0.0101
                                                                                       
  Regression for Rounds 4-7: y=-0.00436x + 34.259, R^2^ = 0.000174                     Regression for Rounds 4-7: y=-0.105x + 139.755, R^2^ = 0.0269
                                                                                       
  Regression for Rounds 1-3: y=-0.0215x + 39.768, R^2^ = 0.00401                       Regression for Rounds 1-3: y=-0.0604x + 132.429, R^2^ = 0.00873

  ![](media/image61.tiff){width="3.3848884514435698in" height="2.424073709536308in"}   ![](media/image62.tiff){width="3.4669466316710413in" height="2.450926290463692in"}
                                                                                       
  Regression (Undrafted): y=0.00122x + 4.0682, R^2^ = 0.00675                          Regression (Undrafted): y=-0.00239x + 7.863, R^2^ = 0.00734
                                                                                       
  Regression (Rounds 4-7): y=0.0000281x + 4.306, R^2^ = 0.0000026                      Regression (Rounds 4-7): y=0.00224x + 6.666, R^2^ = 0.00482
                                                                                       
  Regression for Rounds 1-3: y=0.00261x + 3.654, R^2^ = 0.0250                         Regression (Rounds 1-3): y=-0.000849x + 7.375, R^2^ = 0.000828
  -------------------------------------------------------------------------------------------------------------------------------------------------------------------------

***Fullbacks***

  ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------
  ![](media/image63.tiff){width="3.616323272090989in" height="2.589815179352581in"}     ![](media/image64.tiff){width="3.5421205161854767in" height="2.5566491688538933in"}
                                                                                        
  Regression (Undrafted): y=0.00387x + 3.854, R^2^ = 0.163                              Regression for Undrafted: y=0.101x + -3.712, R^2^ = 0.0724
                                                                                        
  Regression (Rounds 4-7): y=0.00416x + 3.782, R^2^ = 0.127                             Regression for Rounds 4-7: y=0.0991x + -1.0946, R^2^ = 0.0549
                                                                                        
  Regression (Rounds 1-3): y=0.00315x + 3.940, R^2^ = 0.0721                            Regression for Rounds 1-3: y=0.276x + -43.930, R^2^ = 0.1945
  ------------------------------------------------------------------------------------- -------------------------------------------------------------------------------------
  ![](media/image65.tiff){width="3.643765310586177in" height="2.575926290463692in"}     ![](media/image66.tiff){width="3.616961942257218in" height="2.5768514873140855in"}
                                                                                        
  Regression (Undrafted): y=0.00230x + 31.419, R^2^ = 0.000103                          Regression (Undrafted): y=-0.0659x + 127.909, R^2^ = 0.0241
                                                                                        
  Regression (Rounds 4-7): y=-0.0507x + 44.683, R^2^ = 0.0556                           Regression (Rounds 4-7): y=-0.145x + 145.643, R^2^ = 0.136
                                                                                        
  Regression (Rounds 1-3): y=0.0150x + 28.764, R^2^ = 0.00241                           Regression (Rounds 1-3): y=-0.0306x + 120.253, R^2^ = 0.00312

  ![](media/image67.tiff){width="3.3201388888888888in" height="2.3777023184601926in"}   ![](media/image68.tiff){width="3.394212598425197in" height="2.399507874015748in"}
                                                                                        
  Regression (Undrafted): y=0.00405x + 3.403, R^2^ = 0.0960                             Regression (Undrafted): y=0.0100x + 4.877, R^2^ = 0.151
                                                                                        
  Regression (Rounds 4-7): y=0.00462x + 3.251, R^2^ = 0.0720                            Regression (Rounds 4-7): y=0.00722x + 5.659, R^2^ = 0.0545
                                                                                        
  Regression (Rounds 1-3): y=-0.000285x + 4.327, R^2^ = 0.000550                        Regression (Rounds 1-3): y=0.0212x + 2.134, R^2^ = 0.310
  ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------

In order to understand how to slice our data and find
patterns/relationships between different variables, we really have to
dig deep into the key question we want to ask and what the stakeholders
find important from a heuristic level. By integrating both the domain
knowledge of both football experts and inferential techniques, we can
begin to extract more relevant insights.

The key questions we are trying to answer are the following:

1.  How strong is the relationship between athletic ability, measured by
    the drills at the NFL Combine, and how high the player is drafted?

2.  What is this relationship, and can it be modeled?

Any football expert can attest that context is incredibly important in
analyzing the Combine measurable. For example, running a 4.8 second
40-yard dash is either extremely bad or extremely good, depending on the
context. With that in mind, the most important dimensions for providing
context to the combine performances are the player’s position (QB, RB,
DB, etc) and his weight. It makes intuitive sense to group players by
their positions, since each group usually comprises of players that
carry the same role for their team, so their physical attributes,
abilities, and skillsets group together. In addition, even for the same
players within the same position, a very important consideration is the
player’s weight. Expectations for speed, agility, and strength drills
are vastly different players with weight differences of 20 pounds, so
the burning question is, by how much?

To address this question, we split this exercise into two parts. First,
we run a regression for each drill against weight of all the players to
get a good sense of the relationship. Next, we can investigate further
into each position group and run the same regression so that we can
extract more detailed insights into each, and possibly find new trends
or patterns. Furthermore, we can understand the spread of this data by
finding the R-squared of each of these regressions. Lastly, as a point
of reference, we will find the average weights of each position, then
use these to find the average measurable for each combine drill for each
group (Rounds 1-3, 4-7, and Undrafted) by plugging this value into the
regression equations. This will serve as a general sense of where each
of these groups stands.

The results confirm the original hypotheses that vertical leap and broad
jump have negative correlations with weight, while bench press, 40-yard
dash, shuttle, and 3Cone are positively correlated with weight. The
ranges of the R-squared for each exercise, in descending order, are:

1.  40-Yard Dash: 0.68 – 0.75

2.  3Cone: 0.53 – 0.61

3.  Shuttle: 0.52 – 0.59

4.  Broad Jump: 0.40 – 0.48

5.  Bench Press: 0.36 – 0.42

6.  Vertical Leap: 0.32 – 0.41

Furthermore, we can also confirm our intuition that performance at the
NFL Combine has a fairly strong relationship with which draft group that
the prospect gets selected in (Rounds 1-3, 4-7, or Undrafted). We can
see this by observing the slopes and orders of the regression lines. For
exercises where a lower value is considered a better performance
(40-Yard Dash, 3Cone, and Shuttle), the regression line for Rounds 1-3
is predictably below the other two, with Rounds 4-7 above it, then
Undrafted on top. These regression lines form the reverse order when the
higher value is considered a better performance (Broad Jump, Bench
Press, and Vertical Leap).

The position with the highest R-Squares are following for each exercise:

1.  40-Yard Dash: Defensive Linemen

2.  Bench Press: Wide Receivers (not including Quarterbacks because of
    the very low sample size)

3.  Vertical Leap: Defensive Linemen

4.  Broad Jump: Defensive Linemen

5.  Shuttle: Defensive Linemen

6.  3Cone: Defensive Linemen

When looking at the graphs, an interesting pattern that we see is that
the heaviest position groups (Offensive Linemen, Defensive Linemen, and
Tight Ends) generally have the steepest regression line slopes and
highest R-Squared values, while the lightest groups (Wide Receivers and
Defensive Backs) generally have the flattest slopes and higher variance
(we can see these trends visually in the graphs, and numerically in the
swarmplots below). There are two major reasons why this is generally the
case. The first is that weight, at the upper end, weighs down on
performance of combine drills more precipitously than at the lower end.
For example, the 20 pounds difference from 250 lbs to 270 lbs weighs
down a lot more on the vertical leap than from 180 to 200 lbs. Second,
some positions have high statistical variance based on the nature of the
position. For example, a defensive back with weak combine measurables
can easily overcome such deficiencies with other intangible skills such
as a strong ability to catch the ball or tackle. However, for the large
bodies in the trenches, there is much less of a need for these
specialized skills and much more of a need for them to fit a
“prototypical build.” We can see more separation in the regression lines
of each Round group than that of other position groups.

To give life to these numbers, swarmplots make the most sense. I created
a new CSV file with the following columns – Position, R\_Squared, Drill,
Rounds, and Slope. The R\_Squared and Slope columns have the Undrafted,
Rounds 1-3, and Rounds 4-7 values for each position group and drill. The
swarmplots draw from the data from this spreadsheet, and give us the
freedom to group by position.

  ![](media/image69.tiff){width="3.5996642607174105in" height="2.654676290463692in"}    ![](media/image70.tiff){width="3.3913396762904635in" height="2.779676290463692in"}
  ------------------------------------------------------------------------------------- ------------------------------------------------------------------------------------
  ![](media/image71.tiff){width="3.605487751531059in" height="2.5820548993875767in"}    ![](media/image72.tiff){width="3.420501968503937in" height="2.449577865266842in"}
  ![](media/image73.tiff){width="3.702846675415573in" height="2.2749300087489064in"}    ![](media/image74.tiff){width="3.393928258967629in" height="2.3999300087489064in"}
  ![](media/image75.tiff){width="3.6823053368328957in" height="2.5315857392825896in"}   ![](media/image76.tiff){width="3.4628007436570427in" height="2.447995406824147in"}
  ![](media/image77.tiff){width="3.7517158792650918in" height="2.611955380577428in"}    ![](media/image78.tiff){width="3.5254713473315835in" height="2.492299868766404in"}
  ![](media/image79.tiff){width="3.7380818022747158in" height="2.6024617235345584in"}   ![](media/image80.tiff){width="3.500851924759405in" height="2.47489501312336in"}

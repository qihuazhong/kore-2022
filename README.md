# About

This is the 4th place winning solution for the [Kore 2022 simulation](https://www.kaggle.com/competitions/kore-2022-beta/overview), a coding competition sponsored by Google.

It is a rule-based agent, forked from the [awesome bot](https://github.com/w9PcJLyb/kore-beta-bot) with major strategic changes and code re-rewrites.

> :warning: **Warnings**
>
> - This is an archived project.
>
> - Codes include many quick-and-dirty implementations to get quick results within the competition deadline
>
> - This repo include legacy codes that tried to use reinforement learning to train the agent, but dropped and not used at the end.

# Winning strategies

## Gaming strateties

A high-level 📝[write up](https://www.kaggle.com/competitions/kore-2022/discussion/340157) can be found here.

## Coding/computational strategies

Since computation time is limited to 3 seconds/step, and the number of actions is almost unbounded, efficient exploration of different actions become crucial. A few important features make this agent stand out in the competition.

- **Re-writing game logics into vectors/matrics operations with Numpy.**
- Pre-compute candidate routes.
- Heavy use of LRU caching.

# Developement History

TODO

- [ ] Episode 39048741: Don't absorb allied suicide attacker (step 174 (11, 7))

- [ ] Episode 38966269: Proactive rescue fleet (step 92)

  - Epi 39699623 step 161

- [ ] Episode 39715335: step 151 defend multiple wave of attacks

- [ ] Episode 41911001 step 155, waves of reinforcement need!

**0712b**

**0712a**

- [ ] consider absorbing allied ships in roundtrip attack for up to patience 3
- [ ] recapture shipyards

**0711c**

- [x] allow shipyard capturer to sustain damage

**0711b**

- [x] Fixed skip early expansion bug

**0711a**

- [x] Defend shipyard improvement

**0709a**

- check if route is safe for suicide attacker
- allow negative number in net power calculation
- skip early expansion

**0706a**

> validations: 26:13 against `0704a`
> validations: 37:13 against `0702a`

**0704a**

- [x] use pessimistic power estimation for rolling power (for expansion)
- [x] roundtrip attack with absorbing allied fleet
- [x] tighten criteria for expansion
- [x] split ships for mining for the next TWO steps

> validations: 20:10 against `0702a`

**0702**

- [x] adjust expansion criteria: Scale factor increase to 5
- [x] location selection, do not assign points to the closest shipyard

**0629a**

- [x] Fix spawning calculation (use available kore instead of kore)

> validations: 39:11 against `0627a`

**0627a**

- [x] adjust expansion criteria
  - only expand when takeover risk is less than 7
- [x] adjust kore mining value dynamically
- [x] Integrated attack&mining only consider points near attacking target
- [x] If need shipyards, wait for up to two steps to accumulate enough ships
- [x] expansion locations must be distance 4 or further
- [x] shipyard capture distance = 9

> validations: 17:7 against `0625a`

**0625a**

- [x] Episode 39322999: Integrate roundtrip attack into mining (step 251)

> validations: 18:8 against `0623a`

**0620c**

- prioritize collision attack with patience over immediate adjacent attack

**0620b**

- Fix supply depot mining destinations

**0620a**

- Fix mining parameters
- Fix reinforcement needs calculations

**0618c**

- Mining speed improvement: ignore mining targets that are not safe for sure
- net power calculation:
  - consider both optimistic and pessimistic situations

**0618b**

> validations: 40:10 against `0618a`

- adjust kore mining score (prioritize kore at the frontier and in enemy controlled area)
- max distance = 13 when there are only 2 shipyards

**0618a**

> validations: 37:13 against `0615a`

- tighten expansion criteria (0.55 ratio)
- Don't greedy spawn when defensive power is weak (opportunity for long mining route)
- Enhance shipyard reinforcement
- Enhance intercepting suicide attackers

**0617a**

> validations: 20:7 against `0615a`

- limit potential spawn according to incoming kore when calculating estimated future ship counts
- improve command calculation speed
- fix length command calculation

**0614c**

- try to predict whether op shipyard will launch ships (launch when no incoming in the next step and there is enough kore for max spawn)
- wait for next step to mine if it is more profitable (1.4x kores covered)

**0614b**

> validations: 9:1 against `0530c`

- Don't build in interior area (when 4 closest sys are all friendly)
- supply depots dont scale down mining

**0614a**

> validations: 34:16 against `0530c`

- allow friendly intercept with shipyards or fleet for the route to convert
- fix patience for roundtrip attack
- small fix in find non intercept routes(max_t)
- estimate shipyard ship count -> change patience to 2.5

**0609b**

- Do not rescue suicide attacker
- increase scale factor for expansion 3 -> 4

**0609a**

- fixed patience in roundtrip attack

**0604a**

- changed net power calculation

[//]: # '  - limit the power of shipyards to its controlled points)'
[//]: # '  - patience for opponent set to 3'

- attack close shipyards first

[//]: # '- wait for the next step to mine, if the kore covered per turn is significantly higher'

**0530c**

> validations: 36:14 against `0530a`

- scale down mining fleet only when doing so would not stop the fleet from mining more profitable dangerous route
- remove duplicates from route candidates

**0530b**

> validations: 44:6 against `0527c`

- bug fix: enable suicide attack in defend mode if the attack target is incoming hostile fleet

- allow evacuate (roundtrip attack)

- defend future shipyard

- bug(ish) not fixed: sending extra fleets for defending shipyards (seems to have huge positive effect)

**0530a**

> validations: 35:15 against `0527c`

- allow friendly collision when mining
- enhanced mining
- consider all route candidates in finding suicide routes
- add shipyard ship count to the calculation of max_fleet size
- fix roundtrip attack patience

**0527c**

- allow patience for roundtrip attack
- improve net power calculation speed
- bug fix: requiring more ship count than needed for a oneway trip when capturing shipyard

**0527b**

> validations: 38:12 against `0527a`

- adjust expansion location selection:
  - max distance (6, 9)

**0527a**

> validations: 27:13 against `0525b`

- allow absorbing allied fleet when rescuing

> validations: 37:23 against `0525b`

- bug fix: calculation of final_ship_count
- improve intercept suicide attacker

**0525b**

> validations: 31:19 against `0523a`

- scale down mining fleet
- intercept suicide attacker (in roundtrip attacks)

**0525a**

- joint defend fleet
- joint shipyard attack
- shipyard prioritize spawning under defend mode
- fixed a bug when rescue mission uses suicide route
- deduct convert cost from fleets with C command when calculating total ship counts

> validations: 34:16 against `0523a`

- replace frontier risk with takeover risk
- bug fix: spawning ships forever but not attacking shipyards
- **0523a**

> validations: 18:12 against `0519a`

- use sparse matrix calculation
- added patience param for capturing shipyards
- -expansion location more conservative

**0519a**

- use lru_cache
- updated expected kore calculation

**0518a**

- include fleet damage positions in the control map calculation
- more conservative for defensive control map
- allow zigzag route for suicide attack

**0516b**

- minor fixes for defending shipyard
- add (quasi) suicide attack for rescuing ships
- consider attack en_route when trying to capture shipyard

to be implemented and tested: - check is_protected_route for expansion

**0516a**

> validations: 37:13 against `0510a`

- Fixed max route len for mining routes
- alternate return route for mining routes
- preload return routes
- mining speed optimization

**0511a**

> validations: 17:13 against `0510a`

- Added delay attack

**0510a**

> validations: 31:19 against `0509b`

- check if roundtrip attack route is protected up to target_time + 5

**0509b**

> validations: 10:4 against `0509a`

- relaxed expansion requirements: when ships/shipyards > 100

**0509a**

- adjust mining return destination based on frontier risk, where
  `frontier risk` calculates the number of opponent shipyards in 10 Manhattan distance.

  > validations: 32:18 against bot `0508c`

**0508c**

> validations: 7:1 against bot `0508b`

- increase mid and late game mining distances

**0508b**

> validations: 38:12 against bot `0508`

- add place holder for future shipyard (so that it is considered as return locations for mining fleets).

**0508**

- check control map for roundtrip attack

# Project VULCAN

  ![vulcan](https://github.com/ERC-BPGC/project-vulcan/assets/64892362/435007e0-b9ac-4111-8b91-9db8e8976c05)

A project in which we aim at making a humanoid robot's facial structure that can interact with humans and have some simple skills and interactions.

## Usage of this project

*	The main use of this robot is to be an assistant just like Alexa and other assistants... So, then what is the difference? well... this robot will give a human interactive and fell to the answers it... so it feels as if you are conversating with a human and not a speaker ... 
*	Planning to use a good camera with a decent FOV as we need to track the humans around it so the precision will be needed... 


## Major Features

We aim this bot to have facial movements like humans and also have expressions. 

Elaborating:

```
*	Eyeballs moving and having eye contact with the person to be interacted.
*	Mouth moving when the robot speaks.
*	Eyebrow movements.
*	Neck movement.
*	Hearing and speaking.
*	2 cameras in the eyes for vision.
*	Analysing what the person has spoken and given a sensible reply... trying to integrate with google, Wikipedia and other platforms that can give a good answer for a question.
```


### Going deeper into the motions

```
*	Each eyeball having independent motion in 2 axes.
*	Each eyeball having an eyelid that can open and close.
*	Eyebrows that can up-down to add to the expressions.
*	Mouth that can bring about expressions with lips.
*	Neck motion to move the entire system about 2 axes.
```

## Sensing

*	Camera for human detection and other things in the environment.
*	Microphones for sound detection and locating the sound source.

## Subsystems

### Mechanical
*	[Eye, eyelid and eyebrow subsystem](https://www.youtube.com/watch?v=uqxhR49N3ws): Eyeballs non symmetric motion with cameras in each. Eyelids opening and closing. Eyebrows for motion.
*	[Mouth subsystem](https://www.youtube.com/watch?v=Ke2lJfY4haM):  . Movement while delivering words and also depicting emotions through lips.
*	[Neck subsystem](https://www.youtube.com/watch?v=GJRW8hP-Jcs):  . 360-degree motion in horizontal plane and up-down in vertical plane.

### Coding

*	Movement of all mech subsystems: Coordination of all servos and other actuators to bring about emotions and other functions like talking.
*	Hearing and Speech production: Speech recognition, processing it and delivery of words and emotions to the above system. NLM, NLP. 

## Limitations
*	Cannot move physically, just creating the structure above the neck.

## The future of this project?

Well... this is probably the 1% out of the 100 that we can do with... can start another project with bi-paddle motion and mount this on top of it?? and with the robotic arms?? or just the robotic arms??? well that quite open for now... but this is a crucial step as we will be making the so called "head".

## Present existence of similar projects:

*	At present the best example of such a project is Sophia robot, kismet, Ameca robots ... we aim to reach closer to that by this project...

*	However, our robot just comprises of the face but still, it can serve the purpose we aim to make it for... 

*	We have seen many big companies build humanoid robots but a robot that can move and come to use in daily life but starting this at college level would be amazing...


## Built With

* [Python](https://www.python.org/) - The language used
* [ROS](https://www.ros.org/) - Environment used
* [OPENCV](https://opencv.org/) - Used for vision

## Project Leads

* [Parth Shah](https://github.com/Parth-Shah-Tool-Kit) 
* [Ritwik Sharma](https://github.com/Maker-Rat) 

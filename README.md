# Contactless password entering interface

### Tech Stack

- Python

### Installation

run the command `bash libs.sh`

### Description

There are many situations where we need to type in our passwords or confidential data like our ATM pins. Be it using a keyboard or a touch interface. Both, involves the contact of the user with the device. This is unavoidable in cases where u need to type in a pin at an ATM. Surface transmission plays a very big role in the spread of the Corona virus. To overcome this we propose a contactless virtual keyboard that can be controlled with the human eye.

Currently either voice inputs or touchscreens are being used to enter passwords. Besides being contactless to reduce the spread of the virus below are a few other advantages of the proposed solution over the current existing solutions.
It is more effective than giving in voice input as sensitive information can be listened by someone close by.
It is safer than a touchscreen, both in terms of contact and security. The typed data cannot be retrieved using smudges as in the
case of a touchscreen.
It is economically feasible as well. Although it requires an additional camera ,the current touchscreens can be replaced with normal screens. This is a cheaper alternative.
It will be an effective solution for the disabled even when the pandemic comes to an end. Currently the disabled are required to have a companion who would enter the password for them. A few systems accepts voice inputs. Neither of these solutions gives them the privacy that an abled person has. The proposed solution will give them that

The keyboard is controlled with the help of the eye.

The functionalities of the keyboard can be controlled by the :

- Position of the iris (direction whtre the user is looking at).
- Blinking of the eye.

### Libraries

- OpenCV
- dlib

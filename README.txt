 *******************************
*   Delwende Eliane Birba       *
*           EIT                 *
*       Polytech Nice Sophia    *
 *******************************
Machine Learning and Computer Vision:
Training letter: A,B,C,Y
Recognition rate: train = 100.0%, test = 90.0%

 -> Work done :

This Project have been done in 3 Step

Step 1


The first step was to make the face detection algorithm more efficient.
We can discern 2 main stages:
- the cleaning of the original code: indeed, in the code provides, many things
were not useful for the project. The goal is to detect a single face
by video capture, the part dealing with image processing was deleted.
Similarly, the for loop that ran through the face array was replaced by a simple if,
because we detect only one face here.

- The improvement of the algorithm. To implement this enhancement, adding a new
boolean faceDetect was needed. It thus allows, during the first look of face,
to browse the entire video capture looking for the face (as well as if you lose it to a
given time during use). Once detected, we fix a search rectangle
around this face: this will be our search area. The bollean goes to true. This search area will allow greatly reduce the detection time, because we will work in a zone reduced to
this rectangle.

Step 2
The goal was to save in a text file pixel values, corresponding to 16x16 pixels grayscale imagers. Each
thumbnail is a letter in sign language.
All this in order to implement the neural network later.

The steps to achieve this:
 - use of the Facedetect algorithm to detect the face and calculate
   the histogram of the skin;
 - once this detection is done, we do not use Facedetect either to follow the
   face, but tracking the CamShift algorithm (which uses the histogram);
 - we "mask" * the face by reducing the search area from the hand to the part
   left of the image (to detect the right hand);
 - thanks to the histogram, we can find the hand on the part of the remaining image
   (with the backproj, which calculates the area of probability of presence of skin on
   the image) ;
 - we widen the area a little around the hand, so as not to lose part (tips
   of fingers see phalanges), which could affect the quality of the recording
   letters ;
 - a letter of the sign language alphabet is carried out with the hand and one supports
   the corresponding keyboard key to save it to a text file
   (named letter.txt).

Step 3
In this rendering of the project, it was necessary to adapt the camshift / facedetect code with the code letter_recog to recognize letters of the sign language, made with the hand.
We chose here to recognize 4 relatively distinct letters: A, B, C and Y,V.
In the letter_recog code, only the mlp algorithm has been retained.
The topology chosen in letter_recog is a 3-layer, 8-neuron topology.
In the camshift / facedetect code, we have retrieved the function to load the file
learning (we added hard in the code to load the file named letter.xml
at the start of the program).
As for the detection, it is done with the predict method. This function is called
on the loaded learning file. The result of this call is a double, which we can
display on the terminal as a letter of the alphabet after conversion.

-------------------------------------------------------------------------------------------------------
--> How to compile the code?
./build_all.sh

 -> How to execute the code?
1)First run letter_recog with:
./letter_recorg -data= shuffled.txt -save=letter.xml -mlp

2)Launch facedetect:
./facedetect

To detect a letter, we reproduce with the hand the sign then press SPACE to display it
on the terminal.

Note: Letter detection has been most effective after you have mixed the lines of the
data (letter.txt) with the unix shuf command.

Shuf letter.txt -output=shuffled.txt

3)You can also write file  letter.txt and train again
Launch the facedetect do the sign with the hand and click the keyboard corresponding to this sign.
You can repeat to have many data;
After that you can repeat 1 and 2 to test 

NB: --do not type the terminal (to write the letter) right after launched facedetect, we are reading from keyboard not from terminal
    --do not use capital letter



-------------------------------------------------------------------------------------------------------

 -> Analysis :
The light intensity in the room can have an impact
With regard to the network topology:
- Basically, the program implements a 2-layer network topology each composed of 100 neurons.
  In our case, we will only be interested in a small number of letters. We can therefore reduce this value:
  here, the recognition is optimized for a topology of 3 layers with 8 neurons each.
  Note that the detection of letters goes quite well with this topology (test rate at about 90%).
- If we add too many layers, the recognition rate is no longer 100% and the program does not recognize almost
  some letters. The notion of "too many layers" depends on the number of neurons on each layer. For example,
  with the topology 2 layers, 4 neurons on each layer, the recognition rate is no longer 100% from 3
  layers. On the other hand, if we put 50 neurons per layer, we can add a third layer and keep
  a recognition rate of 100%. Adding layers increases learning time, while
  like increasing the number of neurons on each layer.
- For a single layer with 800 neurons, the calculation time increases sharply (we go from about ten seconds
  about 2 minutes for learning). In addition, the program detects letters very badly and sometimes returns
  even letters that are not learned by the program. Single layers with a large number of neurons
  give, in our case, very bad detections of letters.
- The topology limit that does not lower performance is the topology of 3 layers of 20 neurons. Beyond,
  we note either a bad detection of the letters, or a longer computation time (with a result of detection
  similar).
- There is an ambiguity for the recognition of the letters A and Y: this is due to the fact that the detection window
  truncates the hand at the little finger. To support this finding, we do a reconnaissance test
  for these two letters. The test on the A gives a test result of 41.0% and the Y gives a recognition rate
  of 20.6%. It will be necessary to redo the acquisitions for the letter Y after having resized the rectangle around
  from the hand.
- Regarding the quality of the detection, the acquisitions were made for the most part in the same room.
  The acquisition of the Y was made in another room, with another bright environment. That would explain
  also the poor detection of this letter compared to others during the demonstration

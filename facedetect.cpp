#include "opencv2/core/core.hpp"
#include "opencv2/ml/ml.hpp"

#include <opencv2/core/utility.hpp>
#include "opencv2/video/tracking.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/videoio.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/objdetect.hpp"

#include <cstdio>
#include <iostream>
#include <ctype.h>
#include <fstream>

using namespace cv;
using namespace std;
using namespace cv::ml;







int element_shape = MORPH_RECT;
bool visageDetecte = false;
Rect zoneRech;
Mat image;
template<typename T>
static Ptr<T> load_classifier(const string& filename_to_load)
{
    // load classifier from the specified file
    Ptr<T> model = StatModel::load<T>( filename_to_load );
    if( model.empty() )
        cout << "Could not read the classifier " << filename_to_load << endl;
    else
        cout << "The classifier " << filename_to_load << " is loaded.\n";

    return model;
}


void detectAndDraw( Mat& img, CascadeClassifier& cascade, double scale );

string hot_keys =
    "\n\nHot keys: \n"
    "\tESC - quit the program\n";

static void help()
{
    cout << "\nThis is a demo that reconize the sign you made by your right hand\n"
			"\tThe light intensity in the room can have an impact";
    cout << hot_keys;
}

const char* keys =
{
    "{help h | | show help message}{@camera_number| 0 | camera number}"
};

int main( int argc, const char** argv )
{
	bool backprojMode = false;
	bool selectObject = false;
	int trackObject = 0;
	bool showHist = false;
	Point origin;
	Rect selection;
	int vmin = 10, vmax = 256, smin = 30;
	int tolerance = 10;
	double scale = 1.0;

	
	string cascadeName = "haarcascade_frontalface_alt.xml";

bool firstTime = true;
	Rect reductImg;
Rect boundingTrackBox;

Mat src, dst;

    VideoCapture cap;
    Rect trackWindow;
    int hsize = 16;
    float hranges[] = {0,180};
    const float* phranges = hranges;
    CommandLineParser parser(argc, argv, keys);
    CascadeClassifier cascade;
    Ptr<ANN_MLP> model;
    const string& filename_to_load = "letter.xml";
	Rect newtrackWindow;
    if ( parser.has("help") )
    {
        help();
        return 0;
    }
    int camNum = parser.get<int>(0);
    cap.open(camNum);
    if( !cascade.load( cascadeName ) )
    {
        cerr << "ERROR: Could not load classifier cascade" << endl;
        return -1;
    }
    if( !cap.isOpened() )
    {
        help();
        cout << "***Could not initialize capturing...***\n";
        cout << "Current parameter's value: \n";
        parser.printMessage();
        return -1;
    }
    if( filename_to_load.empty() )
    {
        cout << "Error : could not load xml data." << endl;
        return -1;
    }

    // load xml file to analyze
    model = load_classifier<ANN_MLP>( filename_to_load );

    if( model.empty() )
        return false;

    cout << hot_keys;
    namedWindow( "CamShift Demo", 0 );

    Mat frame, hsv, hue, mask, hist, histimg = Mat::zeros(200, 320, CV_8UC3), backproj;
    bool paused = false;

    for(;;)
    {
        // as long as we have not found a face, we run the facedetect algorithm
        while( !visageDetecte )
        {
            cap >> frame;
            detectAndDraw(frame, cascade, scale);
        }

        if( !paused )
        {
            cap >> frame;
            if( frame.empty() )
                break;
        }

        frame.copyTo(image);

        if( !paused )
        {
            cvtColor(image, hsv, COLOR_BGR2HSV);

            if( trackObject )
            {
                int _vmin = vmin, _vmax = vmax;

                inRange(hsv, Scalar(0, smin, MIN(_vmin,_vmax)),
                        Scalar(180, 256, MAX(_vmin, _vmax)), mask);

                int ch[] = {0, 0};
                hue.create(hsv.size(), hsv.depth());
                mixChannels(&hsv, 1, &hue, 1, ch, 1);

                if ( !firstTime )
                {
                    reductImg = Rect(0,0, frame.cols/2, frame.rows);
                }

                // To start the tracking:
                if( trackObject < 0 )
                {

                    Mat roi(hue, selection), maskroi(mask, selection);

                    // The histogram of the skin is determined during the first facial detection.
                    if( firstTime )
                    {
                        calcHist(&roi, 1, 0, maskroi, hist, 1, &hsize, &phranges);
                        normalize(hist, hist, 0, 255, NORM_MINMAX);
                        firstTime = false;


                    trackWindow = selection;
                    trackObject = 1;

                    histimg = Scalar::all(0);
                    int binW = histimg.cols / hsize;
                    Mat buf(1, hsize, CV_8UC3);
                    for( int i = 0; i < hsize; i++ )
                        buf.at<Vec3b>(i) = Vec3b(saturate_cast<uchar>(i*180./hsize), 255, 255);
                    cvtColor(buf, buf, COLOR_HSV2BGR);

                        // We draw the histogram of the skin in histimg:
                        for( int i = 0; i < hsize; i++ )
                        {
                            int val = saturate_cast<int>(hist.at<float>(i)*histimg.rows/255);
                            rectangle( histimg, Point(i*binW,histimg.rows),
                                   Point((i+1)*binW,histimg.rows - val),
                                   Scalar(buf.at<Vec3b>(i)), -1, 8 );
                        }
                    }    

                }

                calcBackProject(&hue, 1, 0, hist, backproj, &phranges);

              
               
				backproj &= mask;
                
                RotatedRect trackBox = CamShift(backproj, trackWindow,
                                    TermCriteria( TermCriteria::EPS | TermCriteria::COUNT, 10, 1 ));
                Rect newtrackbox  = trackBox.boundingRect();


				// change the backpro to zero
				// To detect hand skin, we first remove head skin
				int begX = newtrackbox.x - (image.cols/20) ;
				int endX = newtrackbox.x + newtrackbox.width + (image.rows/20);
    
    
    
				if( begX < 0 ) 
				{
					begX = 0;
				}
				if( endX > image.cols )
				{
					endX = image.cols;
				} 
				// Erase all the height (neck, hair,E...)
				for( int y = 0 ; y < image.rows ; y++ )
				{
					for( int x = begX ; x < endX; x++ )
				{
						backproj.at<uchar>(y,x) = 0;
				}
				}
				// Threshold backproj
				for( int y = 0 ; y < backproj.rows ; y++ )
				{
					for( int x = 0 ; x < backproj.cols ; x++ )
					{
							if( backproj.at<uchar>(y,x) <= 100 )
							{
								backproj.at<uchar>(y,x) = 0;
							}
							else
							{
								backproj.at<uchar>(y,x) = 255;
							}
					}
				}
				//run again:
				newtrackWindow=Rect(0,0,image.rows,image.cols);
				RotatedRect handBox=CamShift(backproj, newtrackWindow,TermCriteria( TermCriteria::EPS | TermCriteria::COUNT, 10, 1 ));
				

               
                if( trackWindow.area() <= 1 )
                {
                    int cols = backproj.cols, rows = backproj.rows, r = (MIN(cols, rows) + 5)/6;
                    trackWindow = Rect(trackWindow.x - r, trackWindow.y - r,
                                       trackWindow.x + r, trackWindow.y + r) &
                                  Rect(0, 0, cols, rows);
                }

                imshow("hand", backproj(newtrackWindow));

                // show the backproj of the hand:
                if( backprojMode ) {
                    cvtColor( backproj, frame, COLOR_GRAY2BGR );
                    imshow("frame", frame);
                }
                // Draw the ellipse at the level of the face.
                ellipse( image, handBox, Scalar(0,0,255), 3, LINE_AA );
            }
        }

        else if( trackObject < 0 )
            paused = false;

        // If we detect a face, we launch the tracking algorithm
         // (putting trackObject at -1).
        if( visageDetecte )
        {
            selection = zoneRech;
            Mat roi(image, selection);
            trackObject = -1;
        }

        imshow( "CamShift Demo", image );

        // User keyboard to interact with the program:
        char c = (char)waitKey(10);
        // Exit the program with the ESC key.
        if( c == 27 )
            break;

        // Display the letter the user made with his hand by pressing SPACE.
        if ( c == 32 )
        {
            src = backproj(newtrackWindow);
            Mat element = getStructuringElement(element_shape, Size(2 + 1, 2 + 1), Point(1, 1) );
            erode(src, dst, element);
            
            element = getStructuringElement(element_shape, Size(4 + 1,4 + 1), Point(2, 2) );
            dilate(dst, dst, element);

            Mat hand =dst; //backproj(newtrackWindow);

            resize(hand, hand, Size(16,16), 0, 0, INTER_LINEAR);
            hand.convertTo(hand, CV_32FC1);

            Mat img = hand.reshape(0,1);

                    CvMat* mlp_response = cvCreateMat(1, 26, CV_32FC1);

           // we go to predict the matrix 1x16
            try 
            {
                float r = model->predict( img );
                // r contains the result of the prediction
                 // to display a letter and not a duplicate on the terminal:
                r += 'A';
                cout << "You made the letter: " << (char) r << endl;
            } catch(Exception e)
            {
                cout << "Mlp predict exception" << endl;
            }
        }

        // Save the pixel values of the images in a letter.txt file:
        else if ( c  >= 'a' && c <= 'z' )
        {
            src = backproj(newtrackWindow);
            Mat element = getStructuringElement(element_shape, Size(2 + 1, 2 + 1), Point(1, 1) );
            erode(src, dst, element);
            
            element = getStructuringElement(element_shape, Size(4 + 1,4 + 1), Point(2, 2) );
            dilate(dst, dst, element);

            Mat hand = dst;//backproj(newtrackWindow);
            resize(hand, hand, Size(16,16), 0, 0, INTER_LINEAR);

            Mat img = hand.reshape(0,1);
            ofstream os("letter.txt", ios::out | ios::app);
            // We capitalize the letter before saving it in the text file
            c = toupper(c);
            os << c;
            // To avoid spaces and brackets:
            int j;
            for(j = 0; j < 256; j++)
            {
                os << "," << (uint)img.at<uchar>(j) ;
            }
            os << std::endl;
            os.close();
        }

    }

    return 0;
}

// Method that detects a face and draws a circle around it.
// NB: this method is used only once, during the first
// face detection. It comes from the facedetect algorithm.
void detectAndDraw( Mat& img, CascadeClassifier& cascade,
                    double scale )
{
    // Our table of faces, which will contain here only one element,
     // to know our face.
    vector<Rect> faces;

    Mat gray, smallImg;

    cvtColor( img, gray, COLOR_BGR2GRAY );
    double fx = 1 / scale;
    // Reduce the image.
    resize( gray, smallImg, Size(), fx, fx, INTER_LINEAR );
    // Equalize the histogram of the image (in grayscale).
    equalizeHist( smallImg, smallImg );

    cascade.detectMultiScale( smallImg, faces,
    1.1, 2, 0
    |CASCADE_SCALE_IMAGE,
    Size(30, 30) );

    // if we detected a face,
     // draw a circle around it.
    if(faces.size() != 0)
    {
        // only one face is detectable during the course of the image
        Rect r = faces[0];
       // We fix the coordinates of the upper left point
       // the search box compared to the previous one
       // search area.
        zoneRech.x = r.x;
        zoneRech.y = r.y;
        zoneRech.width = r.width;
        zoneRech.height = r.height;

        // We specify that we have not lost the face
        visageDetecte = true;
    }

}





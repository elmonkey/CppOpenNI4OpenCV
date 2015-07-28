/*+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
	Version 1.2 by Carlos Torres <carlitos408@gmain.com> Feb5, 2014
	 -- based on the ARC 1.1 by Isaac Flores (major overhaul)
	 * revision 1.2.1 Feb19,2014
	 * revision 1.2.3 Feb21,2014
		1) Increase the number of orientation from 8 to 9
		2) New/shorter frame labels
		3) Single window to display all modes |rgb||skel||mask||depth|
		4) Control flags
		 * N_angle 		      Total # of view angles (=9). Use 4 for testing and 9 for data collection
		 * Increase_angle     Angular step-size in degrees [0:Increase_angle:360]. Use 360/N_angle
		 * RecordingT     	  Time ( secs) to record for each view angle. Use 2 for testing and 6 for data collection
		 * PauseT             Pause(secs). Allowed for actor reposition.  Use 1 for testing and 2 for data collection
		5) All coordinates in a single [15][3] array
			* projective
			* realworld
		6) hdf5 filename = "mmARC_Device"+KINECT_ID+".h5"
		 * --> group name = <actionname>
		 *     --> table name = <actorname>
			NOTE: HDF5 limitations
			 * if duplicate recording is called, the data will be overwritten, but the allocated space is not reclaimed/extended.
			 * Threfore, current work around is to ignore the rows where "viewframe" that are = 0
			 * However, the dataset/table size CANNOT be increased!!.. so a larger recording will be clipped. 
			 * MANUAL WORKOROUND - use vitables to navigate and delete the table/dataset.
			 * Might not be an issue once the experiments (time,angles,etc) that affect the table size are set.

	Todo:
		1) Calibration(first tracking) + key_entered:
		        uninterrupted recording
		        record even w/o actor in the view
        2) Be able to delete an existing table dataset to release memory. file.h5 -> group -> table
     

	Dependencies:
		1)opencv
		2)openni
		3)hdf5/pytables/vitables (sudo apt-get install libhdf5-serial-dev)
		
    Compile: 
        /ARC_v1.2/src/
        cmake .
        make
      
	Run/Usage: 
	 * ENSURE device is connected!!
	    ./ARC1.2 <actionname> <actorname>
	    e.g., ./ARC1.2 test carlos
	
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++*/
 
//---------------------------------------------------------------------------
// Includes
//---------------------------------------------------------------------------
#include <XnCppWrapper.h>
#include <opencv2/opencv.hpp>
#include "opencv2/highgui/highgui.hpp"
#include <iostream>
#include <fstream> 
#include "convert.h"
#include <H5Cpp.h>
#include <hdf5.h>
#include <boost/filesystem.hpp>
#include "boost/date_time/posix_time/posix_time.hpp"
#include <boost/thread/thread.hpp>
#include <time.h>
//---------------------------------------------------------------------------
// Defines
//---------------------------------------------------------------------------
#define SAMPLE_XML_PATH "../../Config/SamplesConfig.xml"
#define SAMPLE_XML_PATH_LOCAL "SamplesConfig.xml"
#define STEREO_FILE "StereoRectify_IR_BGR.xml"
#define MAX_NUM_USERS 1
//#define MAX_NUM_RECORDED_FRAMES  500  // CT: 200-> NOT USED!
#define RANK 3
//#define ROOT_DIR   "/home/carlos/Documents/OPENNI/ARC_v1.2/data" //Directory: Descriptor&Frames 
#define ROOT_DIR   "../data" //Directory: Descriptor&Frames 
#define ImageType1 "mask"			 // CT: "UserPixels" 
#define ImageType2 "skel"			 // CT: "SkelBGR"
#define ImageType3 "rgb"			 // CT: "BGRImage"
#define ImageType4 "depth" 			 // CT: "DepthMap"
#define LENGTH
#define KINECT_ID   2
#define N_angle     9	         	 // CT: total # of view angles (=9)
#define Increase_angle 360/N_angle	 // CT: view angle increments in degrees
#define RecordingT  5   		     // CT: time (in 5 secs) to record each view angle
#define PauseT      2     			 // CT: transition pause(2 secs) for actor repositionls

#define MAX_NUM_GLOBAL_FRAMES   30*RecordingT*N_angle + 200 // CT: 30fps(max) x RecordingT x N_angle + extra

//---------------------------------------------------------------------------
//Namespaces
//---------------------------------------------------------------------------
using namespace std;
using namespace cv;
using namespace xn;
using namespace H5;
namespace PT = boost::posix_time;
//---------------------------------------------------------------------------
// Globals
//---------------------------------------------------------------------------
xn::Context g_Context;
xn::ScriptNode g_scriptNode;
xn::DepthGenerator g_DepthGenerator;
xn::UserGenerator g_UserGenerator;
xn::ImageGenerator image;
XnBool g_bNeedPose = FALSE;
XnChar g_strPose[20] = "";
//---------------------------------------------------------------------------
// Code
//---------------------------------------------------------------------------

XnBool fileExists(const char *fn)
{
	XnBool exists;
	xnOSDoesFileExist(fn, &exists);
	return exists;
}

// Callback: New user was detected
void XN_CALLBACK_TYPE User_NewUser(xn::UserGenerator& /*generator*/, XnUserID nId, void* /*pCookie*/)
{
    XnUInt32 epochTime = 0;
    xnOSGetEpochTime(&epochTime);
    printf("%d New User %d\n", epochTime, nId);
    // New user found
    if (g_bNeedPose)
    {
        g_UserGenerator.GetPoseDetectionCap().StartPoseDetection(g_strPose, nId);
    }
    else
    {
        g_UserGenerator.GetSkeletonCap().RequestCalibration(nId, TRUE);
    }
}
void XN_CALLBACK_TYPE User_Exit(xn::UserGenerator&, XnUserID nID, void*)
{
  cout <<"User Exit." <<endl;
  g_UserGenerator.GetSkeletonCap().StopTracking(nID);

}

void XN_CALLBACK_TYPE User_ReEnter(xn::UserGenerator&, XnUserID nID, void*)
{
  cout <<"User ReEnter." <<endl;
  
}

// Callback: An existing user was lost
void XN_CALLBACK_TYPE User_LostUser(xn::UserGenerator& /*generator*/, XnUserID nId, void* /*pCookie*/)
{
    XnUInt32 epochTime = 0;
    xnOSGetEpochTime(&epochTime);
    printf("%d Lost user %d\n", epochTime, nId);	
}
// Callback: Detected a pose
void XN_CALLBACK_TYPE UserPose_PoseDetected(xn::PoseDetectionCapability& /*capability*/, const XnChar* strPose, XnUserID nId, void* /*pCookie*/)
{
    XnUInt32 epochTime = 0;
    xnOSGetEpochTime(&epochTime);
    printf("%d Pose %s detected for user %d\n", epochTime, strPose, nId);
    g_UserGenerator.GetPoseDetectionCap().StopPoseDetection(nId);
    g_UserGenerator.GetSkeletonCap().RequestCalibration(nId, TRUE);
}
// Callback: Started calibration
void XN_CALLBACK_TYPE UserCalibration_CalibrationStart(xn::SkeletonCapability& /*capability*/, XnUserID nId, void* /*pCookie*/)
{
    XnUInt32 epochTime = 0;
    xnOSGetEpochTime(&epochTime);
    printf("%d Calibration started for user %d\n", epochTime, nId);
}

void XN_CALLBACK_TYPE UserCalibration_CalibrationComplete(xn::SkeletonCapability& /*capability*/, XnUserID nId, XnCalibrationStatus eStatus, void* /*pCookie*/)
{
    XnUInt32 epochTime = 0;
    xnOSGetEpochTime(&epochTime);
    if (eStatus == XN_CALIBRATION_STATUS_OK)
    {
        // Calibration succeeded
        printf("%d Calibration complete, start tracking user %d\n", epochTime, nId);		
        g_UserGenerator.GetSkeletonCap().StartTracking(nId);
    }
    else
    {
        // Calibration failed
        printf("%d Calibration failed for user %d\n", epochTime, nId);
        if(eStatus==XN_CALIBRATION_STATUS_MANUAL_ABORT)
        {
            printf("Manual abort occured, stop attempting to calibrate!");
            return;
        }
        if (g_bNeedPose)
        {
            g_UserGenerator.GetPoseDetectionCap().StartPoseDetection(g_strPose, nId);
        }
        else
        {
            g_UserGenerator.GetSkeletonCap().RequestCalibration(nId, TRUE);
        }
    }
}


#define CHECK_RC(nRetVal, what)					    \
    if (nRetVal != XN_STATUS_OK)				    \
{								    \
    printf("%s failed: %s\n", what, xnGetStatusString(nRetVal));    \
    return nRetVal;						    \
}


//Function to make Filenames
void GenerateFilename_ImageFrames(string& ActionDir, char* fileName, char* filetypeExt, int frameID,  string &OutputFile ) 
{
  stringstream ss;
  ss << ActionDir << "/" << fileName  <<  "_" << frameID << filetypeExt;
  OutputFile = ss.str();
  ss.flush();
}


void GenerateDirectoryPath( char* Root_Directory,  string &Action, string &Actor, int viewAngle, string &OutputFullPath, string &OutputActionName, string &hdf5Folder)
{
  stringstream ss;
  ss << Root_Directory << "/" << Action << "/" <<Actor << "/"<< Actor << "_angle" << viewAngle*Increase_angle << "_dev" << KINECT_ID;
  OutputFullPath = ss.str();
  ss.str("");
  ss.clear();
  ss <<  Action << "_" << Actor;
  OutputActionName = ss.str();
  ss.str("");
  ss.clear();  
  //ss << Root_Directory << "/" << Action << "/" <<Actor;  
  ss << Root_Directory;// points to ../data;
  hdf5Folder = ss.str();
  //cout <<"hdf5Folder for data collection: " << hdf5Folder << endl << endl;
}


void help()
{
  
  cout <<"Specify directory name for Action" << endl
       <<"Directory name should be in format: ARC_ \"ACTIONNAME\"_SET\"NAME\" " << endl;
}

void createActionDirectory( string ActionDirectoryPath, string Action , string Actor);
void createHDF5File( string ActionDirectoryPath, string ActionName, string Action, string hdf5Folder, H5File* &DataFile, Group* &groupDescriptor, string &Filename2);//, Group* &groupMetaData, int angle );


bool OverwriteQuestionHandle()
{
  char answer;
  cin >> answer;
  if( answer != 'Y' && answer!='N' )
    {
      cout <<"Enter 'Y' or 'N'  " << endl;
      OverwriteQuestionHandle(); 
    }
  else if( answer == 'Y' ){
    return true;
  }    
  else if( answer == 'N' )
    return false;
}

/* function that displays all modes in one window or canvas*/
void displayImages(Mat rgb, Mat depth, Mat mask, Mat skel)
{
    // all images have the same dimensions, different # channels
    int row = rgb.rows;//  height of images
    int col = rgb.cols;//  width of images
    
    //Set appropriate range [0-255] and # of channels (1->3)
    Mat dpt, msk;
    Mat dst;   
    double min;
    double max;
    minMaxLoc( depth, &min, &max); //, Point &minLoc, Point &maxLoc );
    convertScaleAbs(depth, dst, 255/(max-min));
    cvtColor(dst, dpt, CV_GRAY2BGR);
    cvtColor(mask,  msk, CV_GRAY2BGR);

    //Create the array that will contain the images
    Mat img(row, 4*col, CV_8UC3);
    //Create sections to copy the individual images 
    Mat p1 ( img, Rect(0,0,col,row) );
    Mat p2 ( img, Rect(col, 0, col, row) );
    Mat p3 ( img, Rect(2*col, 0, col, row) );
    Mat p4 ( img, Rect(3*col, 0, col, row) );
    // copy to the sections
    rgb.copyTo(p1);
    msk.copyTo(p2);
    skel.copyTo(p3);
    dpt.copyTo(p4);
    // display
    imshow("all images", img);    
    /*
    imshow("rgb", rgb);
    imshow("skel", skel);
    imshow("depth",dpt);
    imshow("mask", msk);
    */
}


int main(int argc, char** argv)
{
  //Local time object
  PT::ptime  time_micro = PT::microsec_clock::local_time();
  cout << "Start time is: " << time_micro << endl << endl;

  //Generate Directory name for Action Set
  if( argc < 3 ) {
    cout << "Error. Not enough arguments. " << endl;
    cout << "Specify: <ActionName> and <ActorName>" << endl;
    return -1;
  }


  string Action = argv[1];
  string Actor =  argv[2];
  string ActionName,  ActionDirectoryPath, temp;
  string hdf5Folder, Filename2;
  stringstream ss;
  
  
  // GenerateDirectoryPath( ROOT_DIR, Action, SetID, ActionDirectoryPath, ActionName );
  // cout << "Action name is :" << ActionName << endl;
  // Make sure Directory does not already exist
  // cout << "Making sure Directory does not already exist..." << endl;
  // sleep(2);
  
  bool RESPONSE = false;
  GenerateDirectoryPath( ROOT_DIR, Action, Actor, 11 , ActionDirectoryPath, ActionName, hdf5Folder );
  if( boost::filesystem::exists( ActionDirectoryPath ) ) 
    {
      cout << "Directory already exists!" << endl << "Do you want to overwrite it? [Y/N] ";
      RESPONSE = OverwriteQuestionHandle();
      if( !RESPONSE )
	{
	cout << "Abort. You chose not to delete the directory" << endl;
	return -1;
	}
      
    }
  /*     
  //Create new Directory 
  else
    {
    createActionDirectory( ActionDirectoryPath );
    }
  */

  //--------OpenNI Initialization ------------------------//
  XnStatus nRetVal = XN_STATUS_OK;
  xn::EnumerationErrors errors;
  const char *fn = NULL;
  if    (fileExists(SAMPLE_XML_PATH)) fn = SAMPLE_XML_PATH;
  else if (fileExists(SAMPLE_XML_PATH_LOCAL)) fn = SAMPLE_XML_PATH_LOCAL;
  else {
    printf("Could not find '%s' nor '%s'. Aborting.\n" , SAMPLE_XML_PATH, SAMPLE_XML_PATH_LOCAL);
    return XN_STATUS_ERROR;
  }
  printf("Reading config from: '%s'\n", fn);
  
  nRetVal = g_Context.InitFromXmlFile(fn, g_scriptNode, &errors);
  if (nRetVal == XN_STATUS_NO_NODE_PRESENT)
    {
      XnChar strError[1024];
      errors.ToString(strError, 1024);
      printf("%s\n", strError);
      return (nRetVal);
    }
  else if (nRetVal != XN_STATUS_OK)
    {
      printf("Open failed: %s\n", xnGetStatusString(nRetVal));
      return (nRetVal);
    }
  
  nRetVal = g_Context.FindExistingNode(XN_NODE_TYPE_DEPTH, g_DepthGenerator);
  CHECK_RC(nRetVal,"No depth");
  
  nRetVal = g_Context.FindExistingNode(XN_NODE_TYPE_USER, g_UserGenerator);
  if (nRetVal != XN_STATUS_OK)
    {
        nRetVal = g_UserGenerator.Create(g_Context);
        CHECK_RC(nRetVal, "Find user generator");
    }
  
  nRetVal  = g_Context.FindExistingNode(XN_NODE_TYPE_IMAGE,image );
  if( nRetVal != XN_STATUS_OK )
    { 
      cout << "Could not find Image node. Check XML file" << endl;
    }
  
  //Structure that hold frame data
  typedef struct KinectFrame{
	int globalframe;
	int viewframe;
	float confidence[15];
	float realworld[15][3];	 //CT 15rows(joint#), 3cols[xyz]
	float projective[15][3]; //CT
	long timeStamp;
	int actorAngle;
	int actorLabel;
	int actionLabel; 
	}KinectFrame;

   
  //---------------------HDF5 File(s) Initialization------------------------//   
  hsize_t  cdims[3];
  cdims[0] = 50; 
  cdims[1] = 60; 
  cdims[2] = 12;  
  hsize_t oneD[] ={1};
  H5File * DataFile;
  Group *groupDescriptor;//, *groupMetaData;

  //Create mem type for struct 
  hsize_t array_dim[] = {15};
  hsize_t array_dim3[2] = {15,3}; //[2]:= 2 dimensions; (row)-> joint#, (cols)->[x,y,z]
  ArrayType arrayFloat(PredType::NATIVE_FLOAT, 1, array_dim);
  ArrayType arrayFloat3(PredType::NATIVE_FLOAT,2, array_dim3); //CT
  
  CompType mKF( sizeof(KinectFrame));
  /*Insert the componenet types at the appropriate offsets*/
  mKF.insertMember("globalframe",     HOFFSET(KinectFrame, globalframe),  PredType::NATIVE_INT);
  mKF.insertMember("viewframe",       HOFFSET(KinectFrame, viewframe),    PredType::NATIVE_INT);  
  mKF.insertMember("confidence",      HOFFSET(KinectFrame, confidence),   arrayFloat);
  mKF.insertMember("realworld",       HOFFSET(KinectFrame, realworld),    arrayFloat3); //CT
  mKF.insertMember("projective[xyz]", HOFFSET(KinectFrame, projective),   arrayFloat3); //CT
  mKF.insertMember("timestamp",       HOFFSET(KinectFrame, timeStamp),    PredType::NATIVE_LONG);
  mKF.insertMember("viewangle",       HOFFSET(KinectFrame, actorAngle),   PredType::NATIVE_INT);
  mKF.insertMember("actionlabel",     HOFFSET(KinectFrame, actorLabel),   PredType::NATIVE_INT);//
  mKF.insertMember("actorlabel",      HOFFSET(KinectFrame, actionLabel),  PredType::NATIVE_INT);//


  //OpenNI Regsitration for User nodes & Skeletal Tracking
  XnCallbackHandle hUserCallbacks, hUserExit, hUserReEnter, hCalibrationStart, hCalibrationComplete, hPoseDetected;
  if (!g_UserGenerator.IsCapabilitySupported(XN_CAPABILITY_SKELETON))
    {
      printf("Supplied user generator doesn't support skeleton\n");
      return 1;
    }
  
  
  XnUInt32 timeout = 1000;
  //  SetQuickRefocusTimeout(timeout);
  nRetVal = g_UserGenerator.RegisterToUserExit( User_Exit , NULL, hUserExit);
  CHECK_RC(nRetVal, "Register to user Enter");
  nRetVal = g_UserGenerator.RegisterToUserReEnter(User_Exit, NULL, hUserReEnter);
  CHECK_RC(nRetVal, "Register to user ReEnter");
     
  nRetVal = g_UserGenerator.RegisterUserCallbacks(User_NewUser, User_LostUser, NULL, hUserCallbacks);
  CHECK_RC(nRetVal, "Register to user callbacks");
  nRetVal = g_UserGenerator.GetSkeletonCap().RegisterToCalibrationStart(UserCalibration_CalibrationStart, NULL, hCalibrationStart);
  CHECK_RC(nRetVal, "Register to calibration start");
  nRetVal = g_UserGenerator.GetSkeletonCap().RegisterToCalibrationComplete(UserCalibration_CalibrationComplete, NULL, hCalibrationComplete);
  CHECK_RC(nRetVal, "Register to calibration complete");
  
  if (g_UserGenerator.GetSkeletonCap().NeedPoseForCalibration())
    {
      g_bNeedPose = TRUE;
      if (!g_UserGenerator.IsCapabilitySupported(XN_CAPABILITY_POSE_DETECTION))
        {
	  printf("Pose required, but not supported\n");
	  return 1;
        }
      nRetVal = g_UserGenerator.GetPoseDetectionCap().RegisterToPoseDetected(UserPose_PoseDetected, NULL, hPoseDetected);
      CHECK_RC(nRetVal, "Register to Pose Detected");
      g_UserGenerator.GetSkeletonCap().GetCalibrationPose(g_strPose);
    }
  
  g_UserGenerator.GetSkeletonCap().SetSkeletonProfile(XN_SKEL_PROFILE_ALL);
  
  nRetVal = g_Context.StartGeneratingAll();
  CHECK_RC(nRetVal, "StartGenerating");
 
  //Structure to save joints
  KinectFrame KF[2][MAX_NUM_GLOBAL_FRAMES]; 
  //Counters
  int globalFrameCount = 1, viewFrameCount = 1, recordFrameCount = 1;
  bool TRACKING = false, STARTED_TRACKING = false;
  int skelStartFrame = 1;
  //OpenCV & OpenNIStructures for Display -------------------------------------------------
  XnMapOutputMode d_map, i_map;
  Mat cv_bgr, cv_depth, cv_rgb;
  Mat skel_BGR,  cv_image;
  //Mat blank_image, blank;
  Mat userImage;
  
  Scalar scalars[15] = { Scalar(0,0,255),    Scalar(255,0,0),     Scalar(0,255,255),   Scalar(255,0,255), Scalar(255,255,255),  
						 Scalar(255,0,255),  Scalar(0,0,0),       Scalar(60,200,0),    Scalar(0,0,128),   Scalar(0,128,0), 
						 Scalar(128,0,0),    Scalar(0,128,128),   Scalar(128,128,0),   Scalar(0,128,255), Scalar(128,0,255)   
                        };
  Scalar YELLOW = Scalar(0,255,255);
  Scalar GREEN  = Scalar(0,255,0);
   //Skeletal Joint Tracking Data structures-----------------------
  XnUserID aUsers[ MAX_NUM_USERS ];
  XnUInt16 nUsers;
  Point3f ALL_JOINTS[15];
  Point2f ALL_JOINTS2D[MAX_NUM_USERS][15]; 
  XnVector3D pt;
  XnSkeletonJointTransformation JOINT[15];
  SceneMetaData pScene[2];			     
  //Strings for Image files
  string userImageFilename, skelBGRImageFilename, BGRImageFilename, DepthMapFilename;
  //Time object
  PT::ptime time_micro_end;
  //FPS
  time_t start, end;
  int counter = 0;
  double secs, fps;

  //Check for calibration pose
  cout <<"Locating User(s)..." << endl;
  if(g_bNeedPose)
    {
      printf("Assume calibration pose\n");
    }
  
  bool DetectedUser = false, startRecording = false;
  int angle = 0;
  int startT = 0, nowT = 0;
  PT::ptime timeSec;
  while(N_angle > angle)//( angle < N_angle ) //CT: 8->9->N_angle
    {
      g_Context.WaitOneUpdateAll(  g_UserGenerator );
      while( !DetectedUser  )
	{

	  g_Context.WaitOneUpdateAll(  g_UserGenerator );
	  //Update number of users
	  nUsers=1; //CT:3->1      
	  g_UserGenerator.GetUsers(aUsers, nUsers);
       
	  for( XnUInt i = 0; i < nUsers; i++)
             {
	    if( g_UserGenerator.GetSkeletonCap().IsTracking(aUsers[i])==TRUE )
	      DetectedUser = true;
             }
	} // while DetectedUser
      if( !startRecording )
	{
	  timeSec = PT::second_clock::local_time();
	  //int secsLeft = 60-  ( (timeSec.time_of_day().total_seconds())%3600)%60; //CT:60->30
	  //cout << secsLeft << " seconds until the next 1/2 minute." << endl;
	  /* CT: commented out: no need to press key while testing
	  cout << "Press ANY KEY follow by ENTER to start recording  ";
	  char key;
	  cin >> key;*/
	  //Wait until next minute
	  timeSec = PT::second_clock::local_time();
//	  int secsLeft = 60 -  ( (timeSec.time_of_day().total_seconds())%3600)%60; //CT:60->30
	  int secsLeft = 10 -  ( (timeSec.time_of_day().total_seconds())%3600)%10; //CT:60->30	  
	  cout << "Wating for " << secsLeft << " secs til the next recording!" << endl;
	  startT = timeSec.time_of_day().total_seconds();
	  
	  while( nowT - startT !=  secsLeft )
	    {
	      g_Context.WaitOneUpdateAll(  g_UserGenerator );
	      image.GetMapOutputMode(i_map);
	      const XnRGB24Pixel * image_map = image.GetRGB24ImageMap();
	      imageMap_to_mat(image_map, cv_rgb,i_map.nYRes ,i_map.nXRes);
	      cvtColor(cv_rgb, cv_bgr, CV_BGR2RGB);
	      Point2f crossV1, crossV2, crossH2, crossH1;
	      crossV1.x = 320;
	      crossV1.y = 220;
	      crossV2.x = 320;
	      crossV2.y = 260;
	      crossH1.x = 300;
	      crossH1.y = 240;
	      crossH2.x = 340;	
	      crossH2.y = 240;
	      line(cv_bgr, crossV1, crossV2, Scalar(50,205,50), 4 );
	      line(cv_bgr, crossH1, crossH2, Scalar(50,205,50), 4 );

	      //waitKey( 10 ); //CT: not necessary for cont. streaming
	      timeSec = PT::second_clock::local_time();
	      nowT =   timeSec.time_of_day().total_seconds();
	      // cout << "time is: " << timeSec.time_of_day() << endl;
	    } // while secsLeft

	  startRecording = true;
	}
      else
	{
	  //---------Create HDF5 file-----------//
	  GenerateDirectoryPath( ROOT_DIR, Action, Actor, angle, ActionDirectoryPath, ActionName, hdf5Folder );
	  if( boost::filesystem::exists(ActionDirectoryPath) ) //&& boost::filesystem::exists(hdf5Folder) )  
	    boost::filesystem::remove_all(ActionDirectoryPath);
	  createActionDirectory( ActionDirectoryPath, Action, Actor );
	  createHDF5File( ActionDirectoryPath, ActionName, Action, hdf5Folder, DataFile, groupDescriptor, Filename2);//, groupMetaData, angle  );
	  cout << "Saving frames to: " << ActionDirectoryPath << endl;
	  
	  //Wait for two seconds
	  timeSec = PT::second_clock::local_time();
	  startT =   timeSec.time_of_day().total_seconds();
	  
	  
	  while( nowT - startT != PauseT ) // PauseT = 2 -> 4
	    {
           //imshow("BGR Camera", cv_bgr);
	      timeSec = PT::second_clock::local_time();
	      nowT =   timeSec.time_of_day().total_seconds();
	    }
		
		
	  //------Get start time for current angle iteration----//
	  timeSec = PT::second_clock::local_time();
	  startT   = timeSec.time_of_day().total_seconds();
	  cout << "Start time for view angle " << angle*Increase_angle << " is: " << timeSec.time_of_day () << endl;

	  //Reset counters
	  //globalFrameCount = 0, recordFrameCount = 0;
	  viewFrameCount = 1;
	  
	  while( nowT - startT != RecordingT ) //CT: 10->5->RecordingT; RECORDING TIME FOR THE CURRENT ANGLE
	    {
	      if( counter == 0 )
	    time(&start);
	  //Update all Nodes
	  g_Context.WaitOneUpdateAll(  g_UserGenerator );
	  // Get time stamp for frame    
	  time_micro = PT::microsec_clock::local_time(); 
	  long microTime = time_micro.time_of_day().total_microseconds();
	  long milTime  = time_micro.time_of_day().total_milliseconds();
	  
	//  cout << "Current MICROtime for frame : " << globalFrameCount << " is : " << microTime << "{  " << time_micro.time_of_day() <<  " } "  << endl;
	  //cout << "Current MILLItime for frame : " << globalFrameCount << " is : " << milTime << "{  " << time_micro.time_of_day() <<  " } "  << endl;
	  KF[0][globalFrameCount].timeStamp = milTime;
	  KF[1][globalFrameCount].timeStamp = milTime;

	  PT::time_duration timeMicro = PT::microseconds( microTime );
	  
	  //Retrieve and Display RGB Image from Kinect
	  image.GetMapOutputMode(i_map);
	  const XnRGB24Pixel * image_map = image.GetRGB24ImageMap();
	  imageMap_to_mat(image_map, cv_rgb,i_map.nYRes ,i_map.nXRes);
	  cvtColor(cv_rgb, cv_bgr, CV_BGR2RGB);
	  // copy to sketelon image
	  cv_bgr.copyTo( skel_BGR );
	  //Retrieve and Display depth map from Kinect 
	  const XnDepthPixel*  depth_map = g_DepthGenerator.GetDepthMap();
	  g_DepthGenerator.GetMapOutputMode(d_map);
	  depthMap_to_mat(depth_map,cv_depth ,i_map.nYRes,i_map.nXRes);
  	  
	  //Update number of users
	  nUsers=1;//CT: 3->1
	  g_UserGenerator.GetUsers(aUsers, nUsers);
	  for( XnUInt i = 0; i < nUsers; i++)    
	    {
	      if(g_UserGenerator.GetSkeletonCap().IsTracking(aUsers[i])==FALSE)
			{
			  //   continue;
			  TRACKING == false;
			}
	      else
			{
			  TRACKING = true;
			}
	      if( TRACKING = true && STARTED_TRACKING == false )
			{
			  skelStartFrame = globalFrameCount;
			  STARTED_TRACKING = true;
			}
	      if( TRACKING = true )
			{
			  //Get 15 Skeletal Joint Data Structures-------------------
			  g_UserGenerator.GetSkeletonCap().GetSkeletonJoint(aUsers[i],XN_SKEL_HEAD,JOINT[0]);                  //HEAD
			  g_UserGenerator.GetSkeletonCap().GetSkeletonJoint(aUsers[i],XN_SKEL_NECK, JOINT[1]);                 //NECK
			  g_UserGenerator.GetSkeletonCap().GetSkeletonJoint(aUsers[i],XN_SKEL_TORSO,JOINT[2]);                 //TORSO
			  g_UserGenerator.GetSkeletonCap().GetSkeletonJoint(aUsers[i],XN_SKEL_LEFT_SHOULDER,JOINT[3]);         //LEFT SHOULDER
			  g_UserGenerator.GetSkeletonCap().GetSkeletonJoint(aUsers[i],XN_SKEL_LEFT_ELBOW,JOINT[4]);            //LEFT ELBOW
			  g_UserGenerator.GetSkeletonCap().GetSkeletonJoint(aUsers[i],XN_SKEL_LEFT_HAND,JOINT[5]);             //LEFT HAND
			  g_UserGenerator.GetSkeletonCap().GetSkeletonJoint(aUsers[i],XN_SKEL_LEFT_HIP,JOINT[6]);              //LEFT HIP
			  g_UserGenerator.GetSkeletonCap().GetSkeletonJoint(aUsers[i],XN_SKEL_LEFT_KNEE,JOINT[7]);             //LEFT KNEE
			  g_UserGenerator.GetSkeletonCap().GetSkeletonJoint(aUsers[i],XN_SKEL_LEFT_FOOT,JOINT[8]);             //LEFT FOOT	    
			  g_UserGenerator.GetSkeletonCap().GetSkeletonJoint(aUsers[i],XN_SKEL_RIGHT_SHOULDER,JOINT[9]);        //RIGHT SHOULDER
			  g_UserGenerator.GetSkeletonCap().GetSkeletonJoint(aUsers[i],XN_SKEL_RIGHT_ELBOW, JOINT[10]);         //RIGHT ELBOW
			  g_UserGenerator.GetSkeletonCap().GetSkeletonJoint(aUsers[i],XN_SKEL_RIGHT_HAND,JOINT[11]);           //RIGHT HAND
			  g_UserGenerator.GetSkeletonCap().GetSkeletonJoint(aUsers[i],XN_SKEL_RIGHT_HIP,JOINT[12]);            //RIGHT HIP
			  g_UserGenerator.GetSkeletonCap().GetSkeletonJoint(aUsers[i],XN_SKEL_RIGHT_KNEE,JOINT[13]);           //RIGHT KNEE
			  g_UserGenerator.GetSkeletonCap().GetSkeletonJoint(aUsers[i],XN_SKEL_RIGHT_FOOT,JOINT[14]);           //RIGHT FOOT
			  
			  // Writing to array:float JointPoints[30][15][3]--------  
			  for( int joint = 0; joint < 15; joint++)
				  {
					//Getting 2D Coordiantes to draw Skeletal on RGB
					pt = JOINT[joint].position.position;
					g_DepthGenerator.ConvertRealWorldToProjective(1, &pt, &pt);
					ALL_JOINTS2D[i][joint] = {pt.X, pt.Y};			   
					//Get Joint confidence 
					KF[i][globalFrameCount].confidence[joint] = JOINT[joint].position.fConfidence;

					for( int XYZ = 0; XYZ < 3; XYZ++)
						{
						  if( i > 2 )
							{
							  cout << "There are more than two Users. " << endl 
								   << "Only saving first two Users Joint Coordinates" << endl;
							  break;
							}
							
							  if( XYZ == 0 ){
							KF[i][globalFrameCount].projective[joint][0]=  pt.X; //CT
							KF[i][globalFrameCount].realworld[joint][0] = {JOINT[joint].position.position.X}; //CT
							  }
							  
							  if( XYZ == 1){
							KF[i][globalFrameCount].projective[joint][1]=  pt.Y; //CT
							KF[i][globalFrameCount].realworld[joint][1] = {JOINT[joint].position.position.Y};
							  }
							  
							  if( XYZ == 2){
							KF[i][globalFrameCount].projective[joint][2]= pt.Z; //CT
							KF[i][globalFrameCount].realworld[joint][2] = {JOINT[joint].position.position.Z};
							  }
						} //for XYZ
					}//for joint
				  
			  if( i == 0 )
				{
				  recordFrameCount++;
				}
			} // END if TRACKING == true
	      
	      
	      if( TRACKING == false ) 
			  {
				for( int joint = 0; joint < 15; joint++)
					{
						  //Getting 2D Coordiantes to draw Skeletal on RGB			 
						  pt = { KF[i][globalFrameCount-1].realworld[joint][0], KF[i][globalFrameCount-1].realworld[joint][1], 
								 KF[i][globalFrameCount-1].realworld[joint][2]};
						  g_DepthGenerator.ConvertRealWorldToProjective(1, &pt, &pt);
						  ALL_JOINTS2D[i][joint] = {pt.X, pt.Y};			   
						  //Get Joint confidence
						  KF[i][globalFrameCount].confidence[joint] = KF[i][globalFrameCount-1].confidence[joint];
						  
						  for( int XYZ = 0; XYZ < 3; XYZ++)
							  {
								if( i > 2 ){
									cout << "There are more than two Users. " << endl 
										 << "Only saving first two Users Joint Coordinates" << endl;
									break;
								  }
								  
								  if( XYZ == 0 ){
									KF[i][globalFrameCount].projective[joint][0]=  pt.X; //CT
									KF[i][globalFrameCount].realworld[joint][0] = {JOINT[joint].position.position.X}; //CT			
								  }
								  
								  if( XYZ == 1){
									KF[i][globalFrameCount].projective[joint][1]=  pt.Y; //CT
									KF[i][globalFrameCount].realworld[joint][1] = {JOINT[joint].position.position.Y};			
								  }
								  
								  if( XYZ == 2){
									KF[i][globalFrameCount].projective[joint][2]= pt.Z; //CT
									KF[i][globalFrameCount].realworld[joint][2] = {JOINT[joint].position.position.Z};			
								  }
							  } // for XYZ
						} // for joint
				} // END if TRACKING == false
	      
	      //Display UserPixels ( Silhouette )
	      userImage = Mat::zeros(i_map.nYRes, i_map.nXRes, CV_8UC1); 
	      g_UserGenerator.GetUserPixels( aUsers[i], pScene[i] );		
	      const XnLabel* userP = (unsigned short* )pScene[i].Data();		
	      for (XnUInt y = 0; y < i_map.nYRes; ++y)
		for (XnUInt x = 0; x < i_map.nXRes; ++x, ++userP)
		  {
		    userImage.at<unsigned char>(y,x) = (*userP) ? 255 : 0; 
		  }
	      
	      
	      for( int i = 0; i<nUsers; i++)
			{
			  //Left limbs are drawn yellow
			  //Right limbs are drawn green
			  //Draw Skeleton on BGR Image----------------------------------	     
			  circle(skel_BGR,ALL_JOINTS2D[i][0],10 , scalars[i], 4);                             //HEAD Circle; CT: 30->10
			  line(skel_BGR, ALL_JOINTS2D[i][2], ALL_JOINTS2D[i][1], scalars[i], 4 );             //TORSO  <--> NECK
			  line(skel_BGR, ALL_JOINTS2D[i][0], ALL_JOINTS2D[i][1], scalars[i], 4 );             //HEAD   <--> NECK
			  line(skel_BGR, ALL_JOINTS2D[i][1], ALL_JOINTS2D[i][3], scalars[i], 4 );             //NECK   <--> L SHOULDER
			  line(skel_BGR, ALL_JOINTS2D[i][1], ALL_JOINTS2D[i][9], scalars[i], 4 );             //NECK   <--> R SHOULDER
			  line(skel_BGR, ALL_JOINTS2D[i][3], ALL_JOINTS2D[i][9], scalars[i], 4 );             //L      <--> R SHOULDERs 
			  line(skel_BGR, ALL_JOINTS2D[i][3], ALL_JOINTS2D[i][4], YELLOW, 4 );                 //L S    <--> L ELBOW
			  line(skel_BGR, ALL_JOINTS2D[i][9], ALL_JOINTS2D[i][10],GREEN , 4 );                 //R S    <--> R ELBOW
			  line(skel_BGR, ALL_JOINTS2D[i][6], ALL_JOINTS2D[i][2], scalars[i], 4 );             //TORSO  <--> L HIP
			  line(skel_BGR, ALL_JOINTS2D[i][12],ALL_JOINTS2D[i][2], scalars[i], 4 );             //TORSO  <--> R HIP
			  line(skel_BGR, ALL_JOINTS2D[i][12],ALL_JOINTS2D[i][13],GREEN, 4 );                  //R HIP  <--> R KNEE
			  line(skel_BGR, ALL_JOINTS2D[i][12],ALL_JOINTS2D[i][6], scalars[i], 4 );             //R HIP  <--> L HIP
			  line(skel_BGR, ALL_JOINTS2D[i][6], ALL_JOINTS2D[i][7], YELLOW, 4 );                 //L HIP  <--> L KNEE
			  line(skel_BGR, ALL_JOINTS2D[i][8], ALL_JOINTS2D[i][7], YELLOW, 4 );                 //L KNEE <--> L FOOT
			  line(skel_BGR, ALL_JOINTS2D[i][14],ALL_JOINTS2D[i][13],GREEN, 4 );                  //R KNEE <--> R FOOT
			  line(skel_BGR, ALL_JOINTS2D[i][4], ALL_JOINTS2D[i][5], YELLOW, 4 );                 //L HAND <--> L ELBOW
			  line(skel_BGR, ALL_JOINTS2D[i][10],ALL_JOINTS2D[i][11],GREEN , 4 );                 //R HAND <--> R ELBOW
			} //for i
	    } // for XnUInt i

	  //Write User Pixels
//	  GenerateFilename_ImageFrames(ActionDirectoryPath, ImageType1, ".png", globalFrameCount, userImageFilename);
	  GenerateFilename_ImageFrames(ActionDirectoryPath, ImageType1, ".png", viewFrameCount, userImageFilename);
	  
	  imwrite(userImageFilename, userImage);
	  userImageFilename.clear();
	  
	  //Write BGR Image with Skeleton Overlay
	  GenerateFilename_ImageFrames(ActionDirectoryPath, ImageType2, ".png", viewFrameCount, skelBGRImageFilename); 
	  imwrite(skelBGRImageFilename, skel_BGR);
	  skelBGRImageFilename.clear();
	  
	  //Write BGR Image
	  GenerateFilename_ImageFrames(ActionDirectoryPath, ImageType3, ".png", viewFrameCount, BGRImageFilename); 
	  imwrite(BGRImageFilename, cv_bgr); 
	  BGRImageFilename.clear();
	  
	  //Write Depth Image
	  GenerateFilename_ImageFrames(ActionDirectoryPath, ImageType4, ".png", viewFrameCount, DepthMapFilename); 
	  imwrite(DepthMapFilename, cv_depth); 
	  DepthMapFilename.clear();
	  
	  //Display ALL Images
          displayImages(cv_bgr, cv_depth, userImage, skel_BGR);       	  
	  	  
	  //Saving global frame#: continues across all views
	  KF[0][globalFrameCount].globalframe = globalFrameCount;
	  KF[1][globalFrameCount].globalframe = globalFrameCount;
	  
	  //Saving view frame #: resets every view 
	  KF[0][globalFrameCount].viewframe = viewFrameCount;
	  KF[1][globalFrameCount].viewframe = viewFrameCount;

	  //CT: save the angle value:
	  KF[0][globalFrameCount].actorAngle = angle*Increase_angle;
	  KF[1][globalFrameCount].actorAngle = angle*Increase_angle;

	  globalFrameCount++;
	  viewFrameCount++;
	  
	  // CT: add all viewsFrameCOunt
	  waitKey(30);
	  
	  //FPS
	  ++counter;
	  if( counter == 30 )
	    {
	      time(&end);
	      secs = difftime(end, start);
	      fps = counter/secs;
	      cout << "FPS (30 frames) is:" << fps << endl;
	      counter = 1; //CT: 0->1, start at frame 1.
	    }
	  timeSec = PT::second_clock::local_time();
	  nowT =   timeSec.time_of_day().total_seconds();
	} //End while RecordingT (for 10 seconds)
 
	  
	  //Write Joint Coordinates to HDF5 File
	  hsize_t oneArrayD[] ={globalFrameCount};
	  DataSpace space1(1, oneArrayD);
	  DataSpace space2(1, oneArrayD);
	  DataSet *datasetFrames_User1, *datasetFrames_User2;
	  DSetCreatPropList propertyList;
	  propertyList.setChunk(RANK, cdims);
	  propertyList.setDeflate(6);
	  
	  /*	CT: modified to save data to a single hdf5 for ALL VIEWS instead of creating a new file for every view*/
	  
	  /* Uncomment to add recording for a second user! Also change define to " MAX_NUM_USERS " to 2.
	  datasetFrames_User2 = new DataSet(DataFile->createDataSet("Kinect_Camera/FrameData_User2", mKF, space2) ); 
	  datasetFrames_User2->write( &KF[1], mKF);  
	  delete datasetFrames_User2;
	  */
	
		 
	  //Increment view angle
	  if(angle != N_angle) //CT: 8->9->N_angle
	    {
	      angle++;
	      cout << "Actor turn left " << Increase_angle << " degrees! for a direction of: " << angle*Increase_angle <<  endl; //45->40->Increase_angle
	    }

	}
    } // while not all the angles


  //Write Joint Coordinates to HDF5 File
  hsize_t oneArrayD[] ={globalFrameCount};
  DataSpace space1(1, oneArrayD);
  DataSpace space2(1, oneArrayD);
  DataSet *datasetFrames_User1, *datasetFrames_User2;
  DSetCreatPropList propertyList;
  propertyList.setChunk(RANK, cdims);
  propertyList.setDeflate(6);
  cout << "Creating the group for angle: "<< angle << endl;
  
		try {  // to determine if the dataset exists in the file
			datasetFrames_User1 = new DataSet( DataFile->openDataSet(Action+"/"+Actor) );
			//DataFile->unlink(Action+"/"+Actor);
			//datasetFrames_User1 = new DataSet(DataFile->createDataSet(Action+"/"+Actor, mKF, space1) ); 			
			cout << " hdf5 dataset was found. Allocated storage remains!." << endl;
			}
		catch( FileIException not_found_error )
		{
			cout << " hdf5 dataset was not found. Creating it!." << endl;
			datasetFrames_User1 = new DataSet(DataFile->createDataSet(Action+"/"+Actor, mKF, space1) ); 
		}  

  datasetFrames_User1->write( &KF[0], mKF);
  delete datasetFrames_User1;
  delete groupDescriptor;
  DataFile->close();  

 cout << "Done writing to HDF5 file for current view" << endl << endl;
  
  // Release openni resources
  g_scriptNode.Release();
  g_DepthGenerator.Release();
  g_UserGenerator.Release();
  image.Release();
  g_Context.Release();
  
} // end main


void createActionDirectory( string ActionDirectoryPath, string Action, string Actor )
{
  stringstream ss;
  string temp;

  //Check if Action root path exist
  ss << ROOT_DIR <<  "/" << Action;
  temp  = ss.str();
  if(  !boost::filesystem::exists( temp) ) 
    {
      boost::filesystem::create_directory( temp );
    }
  ss.str("");
  ss.clear();
  temp.clear();
  
  //Check if Action/Actor Directory exist
  ss << ROOT_DIR <<  "/" << Action << "/" << Actor ;
  temp  = ss.str();
  if(  !boost::filesystem::exists( temp ) ) 
    {
	  cout << "creating the folder: " << temp << endl << endl;
      boost::filesystem::create_directory( temp );
    }
  ss.str("");
  ss.clear();
  temp.clear();
  
  boost::filesystem::create_directory( ActionDirectoryPath );
/*CT: COMMENT THIS GENERATE MODE FOLDERS for the images!*/
  
  //Directories for Image frames
  ss << ActionDirectoryPath;// << "/" << ImageType1 << "_Frames";
  temp = ss.str();
  boost::filesystem::create_directory( temp ); 
  //cout << "0000" << endl;
  ss.str("");
  ss.clear();
  temp.clear();
  //cout << "1111" << endl;
  
} //createActionDirectory


void createHDF5File( string ActionDirectoryPath, string ActionName, string Action, string hdf5Folder, H5File* &DataFile, Group* &groupDescriptor, string &Filename2)
{
  stringstream ss;
  ss << ActionDirectoryPath << "/" << ActionName <<  "_device" << KINECT_ID <<  ".h5";
  string Filename1 = ss.str();
  ss.str("");
  ss.clear();
  ss << hdf5Folder << "/" << "mmARC" <<  "_device" << KINECT_ID << ".h5"; // The group and hdf5file.h5
  
  Filename2 = ss.str();
  ss.str("");
  ss.clear();
  
  // check if file doesnt exists: create it. else open to append data.
  // 	file exists; check if group exists (open), else create the group
  if(  !boost::filesystem::exists( Filename2) ) 
    {
		cout << "hdf5 does not exist, creating it!" << endl << endl;
		DataFile = new H5File(Filename2, H5F_ACC_TRUNC);  
		groupDescriptor = new Group(DataFile->createGroup(Action) );	  
    }
  else // .h5 file does exists, so open it & check that group also exists
	{
		cout << "hdf5 exists, ...opening!" << endl << endl;
		DataFile = new H5File(Filename2, H5F_ACC_RDWR); 
			
		try {  // to determine if the group exists in the file
			groupDescriptor = new Group(DataFile->openGroup(Action) );
			cout << " hdf5 group was found!" << endl;			
		}
		catch( FileIException not_found_error )
		{
			cout << " hdf5 group was not found, creating it!." << endl;
			groupDescriptor = new Group(DataFile->createGroup(Action) );				
		}
	}
} //createHDF5File

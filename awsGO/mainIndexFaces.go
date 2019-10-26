//Indexing faces to a collection

package main

import (
	"fmt"
	"time"

	"github.com/aws/aws-lambda-go/events"
	"github.com/aws/aws-sdk-go/aws"
	"github.com/aws/aws-sdk-go/aws/session"
	"github.com/aws/aws-sdk-go/service/rekognition"
	"github.com/aws/aws-sdk-go/service/rekognition/rekognitioniface"
)

var (
	_rekognitionService rekognitioniface.RekognitionAPI
	collectionID        = "test-video-carlosn" //"NoFaces"
)

// LambdaRequest body of request
type LambdaRequest struct {
	*events.S3Event
	bucketName string
	objectKey  string
	err        error
}


func (req *LambdaRequest) addFaces() error {
	input := &rekognition.IndexFacesInput{
		ExternalImageId:     aws.String("test-video-1.png"), //The ID assigned to the face found
		CollectionId:        aws.String(collectionID),
		DetectionAttributes: []*string{aws.String("DEFAULT")},
		//"MaxFaces":1,
		Image: &rekognition.Image{
			S3Object: &rekognition.S3Object{
				Bucket: aws.String(req.bucketName),
				Name:   aws.String(req.objectKey),
			},
		},

	}

	out, err := _rekognitionService.IndexFaces(input)
	if err != nil {
		return err
	}

	//fmt.Println(out)
	faceRecords := out.FaceRecords
	if len(faceRecords) == 0 {
		fmt.Println("No faces found")
	} else{
		faceConfidence := *out.FaceRecords[0].Face.Confidence
		brightness := *out.FaceRecords[0].FaceDetail.Quality.Brightness
		sharpness := *out.FaceRecords[0].FaceDetail.Quality.Sharpness
		fmt.Println("Confidence of a face: ",faceConfidence," Brightness: ", brightness, " Sharpness: ", sharpness)
	}

	return nil
}

func main() {
	start := time.Now()
	sess, err := session.NewSession(&aws.Config{
		Region: aws.String("us-east-1")},
	)

	if err != nil {
		fmt.Println("Ops: " + err.Error())
		return
	}

	_rekognitionService = rekognition.New(sess)

	request := LambdaRequest{
		bucketName: "prueba-face-recognition",
		objectKey:  "VideoCarlosN/CarlosN.png", //"NoFaces/NoFace.png"
	}

	err = request.addFaces()
	if err != nil {
		fmt.Println("Ops: " + err.Error())
		return
	}
	elapsedTime := time.Since(start)
	fmt.Println("Elapsed time: ", elapsedTime)
}

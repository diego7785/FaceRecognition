//Search faces in the collection based on a video

//Montar lambdas

package main

import (
	"context"
	"fmt"
	"time"

	"github.com/aws/aws-lambda-go/events"
	"github.com/gofrs/uuid"

	"github.com/aws/aws-sdk-go/aws"
	"github.com/aws/aws-sdk-go/aws/session"
	"github.com/aws/aws-sdk-go/service/rekognition"
	"github.com/aws/aws-sdk-go/service/rekognition/rekognitioniface"
)

var (
	_rekognitionService rekognitioniface.RekognitionAPI
	collectionID        = "test-video-joseluis"
	snsTopicArn         = "arn:aws:sns:us-east-1:530395375560:pruebaAmazonRekognition"
	roleARN             = "arn:aws:iam::530395375560:role/pruebaFaceID"
)

// LambdaRequest body of request
type LambdaRequest struct {
	*events.S3Event
	startingTime time.Time
	bucketName   string
	objectKey    string
	err          error
}


func getFaceSearchResult(jobId string) (*rekognition.GetFaceSearchOutput, error){
	sess, err := session.NewSession(&aws.Config{
		Region: aws.String("us-east-1")},
	)

	if err != nil{
		fmt.Println("Ops: ", err.Error())
	}

	_rekognitionS := rekognition.New(sess)

	input := rekognition.GetFaceSearchInput{JobId: &jobId}

	result, err := _rekognitionS.GetFaceSearch(&input)

	if err != nil{
		fmt.Println("Ops: ", err.Error())
	}

	return result, nil
}

func (req *LambdaRequest) startFaceID(ctx context.Context) error {
	input := &rekognition.StartFaceSearchInput{
		ClientRequestToken: aws.String(uuid.Must(uuid.NewV4()).String()),
		CollectionId:       aws.String(collectionID),
		//FaceMatchesThreshold is automatically setted as 80 %
		Video: &rekognition.Video{
			S3Object: &rekognition.S3Object{
				Bucket: aws.String(req.bucketName),
				Name:   aws.String(req.objectKey),
			},
		},
		NotificationChannel: &rekognition.NotificationChannel{
			SNSTopicArn: aws.String(snsTopicArn),
			RoleArn:     aws.String(roleARN),
		},
	}

	out, err := _rekognitionService.StartFaceSearch(input)
	if err != nil {
		fmt.Println("Ops: " + err.Error())

		return err
	}
	JobId := out.JobId
	fmt.Println("JobId: ", *JobId)

	fmt.Println("start GetFaceSearch")
	result, err := getFaceSearchResult(*JobId)
	if err != nil {
		fmt.Println("Ops: ", err.Error())
	}
	
	fmt.Println("GetFaceSearch result acquired, status: ", *result.JobStatus)
	for *result.JobStatus == "IN_PROGRESS"{
		result, err := getFaceSearchResult(*JobId)
		if err != nil {
			fmt.Println("Ops: ", err.Error())
		}
		if *result.JobStatus == "SUCCEEDED"{
			lengthPersons := len(result.Persons)
			aceptacion := int(lengthPersons/2)
			matches := 0
			for i := 0; i<lengthPersons; i++{
				if(len(result.Persons[i].FaceMatches)>=1){
					matches++
				}
			}
			if(matches>=aceptacion){
				fmt.Println("Match")
			} else {
				fmt.Println("No match")
			}
			fmt.Println(result)
			break
		}
	}

	return nil
}



func main() {
	start := time.Now()
	//time.Sleep(time.Second*2)

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
		objectKey:  "VideoJoseLuis/video.mp4",
	}

	request.startFaceID(context.Background())
	fmt.Println("Video processed: ",request.objectKey)
	elapsedTime := time.Since(start)
	fmt.Println("Elapsed time: ", elapsedTime)
}

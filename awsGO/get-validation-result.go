//Input: JobId Output: Match, No match

package main

import(
	"fmt"

	"github.com/aws/aws-sdk-go/aws"
	"github.com/aws/aws-sdk-go/aws/session"
	"github.com/aws/aws-sdk-go/service/rekognition"
)


func getValidationResult(jobId string) error{
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
	return nil	
}
	

func main(){
	getValidationResult("bb6794cd6ffb03696e020a92b01a75ee613fc0ae6ecd0212e69afbc293541340")
}
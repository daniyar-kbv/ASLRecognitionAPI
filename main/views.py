from rest_framework.decorators import api_view, parser_classes
from rest_framework.response import Response
from rest_framework.parsers import MultiPartParser
from django.conf import settings
from main.serializers import PredictionInputSerializer
import numpy as np
import utils
import cv2


@api_view(['POST'])
@parser_classes([MultiPartParser])
def predict(request):
    serializer = PredictionInputSerializer(data=request.data)
    serializer.is_valid(raise_exception=True)
    image = cv2.imdecode(np.frombuffer(request.FILES['image'].read(), np.uint8), cv2.IMREAD_UNCHANGED)
    class_name, bounding_box, score = utils.infer(image, settings.DETECTION_MODEL, settings.CATEGORY_INDEX)
    return Response({
        'class_name': class_name,
        'bounding_box': bounding_box,
        'score': score
    })
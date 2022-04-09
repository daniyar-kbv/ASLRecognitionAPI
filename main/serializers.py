from rest_framework import serializers


class PredictionInputSerializer(serializers.Serializer):
    image = serializers.ImageField()

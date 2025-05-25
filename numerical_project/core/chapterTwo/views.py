from django.shortcuts import render

def chapter_view(request):
    return render(request, 'chapterTwo/chapterTwo.html')

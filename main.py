import streamlit as st
from gtts import gTTS
from bs4 import BeautifulSoup
import nltk

# Initialize NLTK's Vader Sentiment Intensity Analyzer
nltk.download('vader_lexicon')
from nltk.sentiment.vader import SentimentIntensityAnalyzer
sid = SentimentIntensityAnalyzer()

# Function to generate video script based on advanced analysis
def generate_script(subject, language, creativity):
    # Predefined data for topics, titles, and scripts
    predefined_data = {
    'python': {
        'en': {
            'title': 'Python Programming Basics',
            'script': '''
                Introduction:
                    Welcome to our Python programming tutorial! Today, we'll cover the basics of Python, a powerful and versatile programming language used in various fields like web development, data analysis, artificial intelligence, and more.

                    1. What is Python?
                    Python is a high-level programming language known for its simplicity and readability. It allows you to write clear and concise code, making it perfect for beginners and professionals alike.

                    2. Setting Up Python:
                    To start coding in Python, you need to install it on your computer. You can download Python for free from the official website and follow the installation instructions based on your operating system.

                    3. Writing Your First Program:
                    Let's dive into coding! Open a text editor or an Integrated Development Environment (IDE) like PyCharm or Visual Studio Code. Write a simple program to display "Hello, World!" on the screen and run it to see the output.

                    4. Variables and Data Types:
                    In Python, you can store data in variables. There are different data types such as integers, floats, strings, lists, and dictionaries. Learn how to declare variables and perform basic operations like arithmetic calculations and string manipulation.

                    5. Conditional Statements and Loops:
                    Conditional statements like if, else, and elif help you make decisions in your code based on conditions. Loops like for and while allow you to repeat tasks efficiently.

                    6. Functions:
                    Functions are reusable blocks of code that perform specific tasks. Learn how to define and call functions to organize your code and make it more modular.

                    7. Libraries and Modules:
                    Python has a vast ecosystem of libraries and modules that extend its functionality. Explore popular libraries like NumPy for numerical computing, Pandas for data analysis, Matplotlib for data visualization, and TensorFlow for machine learning.

                    8. Handling Errors:
                    Error handling is essential in programming. Learn about try-except blocks to handle exceptions gracefully and prevent your program from crashing.

                    9. Best Practices and Tips:
                    Follow best practices like using meaningful variable names, writing comments, and formatting your code properly to improve readability and maintainability.

                    10. Conclusion:
                    Congratulations on completing our Python tutorial! Keep practicing, explore advanced topics, and build exciting projects to enhance your programming skills.
            ''',
            'duration': 10  # Length of video in minutes
        },
        'fr': {
            'title': 'Bases de la programmation Python',
            'script': '''
                Introduction:
                    Bienvenue dans notre tutoriel de programmation Python ! Aujourd'hui, nous aborderons les bases de Python, un langage de programmation puissant et polyvalent utilisé dans divers domaines tels que le développement web, l'analyse de données, l'intelligence artificielle, et bien d'autres encore.

                    1. Qu'est-ce que Python ?
                    Python est un langage de programmation de haut niveau connu pour sa simplicité et sa lisibilité. Il vous permet d'écrire un code clair et concis, ce qui le rend parfait pour les débutants comme pour les professionnels.

                    2. Configuration de Python :
                    Pour commencer à coder en Python, vous devez l'installer sur votre ordinateur. Vous pouvez télécharger Python gratuitement depuis le site officiel et suivre les instructions d'installation en fonction de votre système d'exploitation.

                    3. Écrire votre premier programme :
                    Plongeons dans le codage ! Ouvrez un éditeur de texte ou un Environnement de Développement Intégré (IDE) comme PyCharm ou Visual Studio Code. Écrivez un programme simple pour afficher "Bonjour, monde !" à l'écran et exécutez-le pour voir le résultat.

                    4. Variables et types de données :
                    En Python, vous pouvez stocker des données dans des variables. Il existe différents types de données tels que les entiers, les nombres à virgule flottante, les chaînes de caractères, les listes et les dictionnaires. Apprenez à déclarer des variables et à effectuer des opérations de base telles que des calculs arithmétiques et des manipulations de chaînes.

                    5. Instructions conditionnelles et boucles :
                    Les instructions conditionnelles telles que if, else et elif vous aident à prendre des décisions dans votre code en fonction de conditions. Les boucles telles que for et while vous permettent de répéter des tâches de manière efficace.

                    6. Fonctions :
                    Les fonctions sont des blocs de code réutilisables qui effectuent des tâches spécifiques. Apprenez à définir et à appeler des fonctions pour organiser votre code et le rendre plus modulaire.

                    7. Bibliothèques et modules :
                    Python dispose d'un vaste écosystème de bibliothèques et de modules qui étendent sa fonctionnalité. Explorez des bibliothèques populaires comme NumPy pour le calcul numérique, Pandas pour l'analyse de données, Matplotlib pour la visualisation de données, et TensorFlow pour l'apprentissage automatique.

                    8. Gestion des erreurs :
                    La gestion des erreurs est essentielle en programmation. Apprenez les blocs try-except pour gérer les exceptions de manière élégante et empêcher votre programme de planter.

                    9. Bonnes pratiques et conseils :
                    Suivez les bonnes pratiques comme l'utilisation de noms de variables significatifs, l'écriture de commentaires et le formatage correct de votre code pour améliorer sa lisibilité et sa maintenabilité.

                    10. Conclusion :
                    Félicitations pour avoir terminé notre tutoriel Python ! Continuez à pratiquer, explorez des sujets avancés et créez des projets passionnants pour améliorer vos compétences en programmation.
            ''',
            'duration': 10  # Length of video in minutes
        },
        'es': {
            'title': 'Conceptos básicos de programación en Python',
            'script': '''
                Introducción:
                    ¡Bienvenidos a nuestro tutorial de programación en Python! Hoy, cubriremos los conceptos básicos de Python, un lenguaje de programación potente y versátil utilizado en varios campos como desarrollo web, análisis de datos, inteligencia artificial y más.

                    1. ¿Qué es Python?
                    Python es un lenguaje de programación de alto nivel conocido por su simplicidad y legibilidad. Te permite escribir código claro y conciso, lo que lo hace perfecto para principiantes y profesionales por igual.

                    2. Configuración de Python:
                    Para empezar a codificar en Python, necesitas instalarlo en tu computadora. Puedes descargar Python de forma gratuita desde el sitio web oficial y seguir las instrucciones de instalación según tu sistema operativo.

                    3. Escribir tu primer programa:
                    ¡Vamos a sumergirnos en la codificación! Abre un editor de texto o un Entorno de Desarrollo Integrado (IDE) como PyCharm o Visual Studio Code. Escribe un programa simple para mostrar "¡Hola, mundo!" en la pantalla y ejecútalo para ver la salida.

                    4. Variables y tipos de datos:
                    En Python, puedes almacenar datos en variables. Hay diferentes tipos de datos como enteros, flotantes, cadenas, listas y diccionarios. Aprende a declarar variables y realizar operaciones básicas como cálculos aritméticos y manipulación de cadenas.

                    5. Sentencias condicionales y bucles:
                    Las sentencias condicionales como if, else y elif te ayudan a tomar decisiones en tu código según condiciones. Los bucles como for y while te permiten repetir tareas de manera eficiente.

                    6. Funciones:
                    Las funciones son bloques de código reutilizables que realizan tareas específicas. Aprende a definir y llamar funciones para organizar tu código y hacerlo más modular.

                    7. Bibliotecas y módulos:
                    Python tiene un vasto ecosistema de bibliotecas y módulos que amplían su funcionalidad. Explora bibliotecas populares como NumPy para cálculos numéricos, Pandas para análisis de datos, Matplotlib para visualización de datos y TensorFlow para aprendizaje automático.

                    8. Manejo de errores:
                    El manejo de errores es esencial en la programación. Aprende sobre bloques try-except para manejar excepciones de manera elegante y evitar que tu programa se bloquee.

                    9. Buenas prácticas y consejos:
                    Sigue buenas prácticas como usar nombres de variables significativos, escribir comentarios y formatear tu código correctamente para mejorar su legibilidad y mantenibilidad.

                    10. Conclusión:
                    ¡Felicidades por completar nuestro tutorial de Python! Sigue practicando, explora temas avanzados y crea proyectos emocionantes para mejorar tus habilidades de programación.
            ''',
            'duration': 10  # Length of video in minutes
        }
    },
    'java': {
        'en': {
            'title': 'Java Programming Basics',
            'script': '''
                Introduction:
                    Welcome to our Java programming tutorial! In this tutorial, we'll cover the basics of Java, a widely used programming language for building applications, especially in the domain of enterprise software development.

                    1. What is Java?
                    Java is a high-level, object-oriented programming language known for its platform independence. It allows you to write code once and run it on any device that supports Java, making it highly versatile.

                    2. Setting Up Java:
                    To start coding in Java, you need to install the Java Development Kit (JDK) on your computer. Download the JDK from the official website and follow the installation instructions based on your operating system.

                    3. Writing Your First Program:
                    Let's dive into coding with Java! Open a text editor or an Integrated Development Environment (IDE) like IntelliJ IDEA or Eclipse. Write a simple program to display "Hello, World!" on the screen and run it to see the output.

                    4. Variables and Data Types:
                    Java supports various data types such as integers, floating-point numbers, characters, strings, arrays, and more. Learn how to declare variables, perform arithmetic operations, and manipulate strings in Java.

                    5. Conditional Statements and Loops:
                    Use conditional statements like if, else, and switch to make decisions in your Java programs. Explore loops like for, while, and do-while to iterate over code blocks efficiently.

                    6. Classes and Objects:
                    Java is an object-oriented language, which means you work with classes and objects. Learn how to define classes, create objects, and access their properties and methods.

                    7. Exception Handling:
                    Exception handling is crucial in Java programming. Learn about try-catch blocks, throw, throws, and finally to manage exceptions gracefully and maintain program stability.

                    8. Packages and Libraries:
                    Java offers a vast ecosystem of packages and libraries for various functionalities. Discover popular libraries like JavaFX for graphical user interface (GUI) development, JDBC for database connectivity, and JUnit for unit testing.

                    9. Best Practices and Tips:
                    Follow best practices like naming conventions, writing clear comments, and organizing your code into reusable modules to enhance code quality and maintainability.

                    10. Conclusion:
                    Congratulations on completing our Java tutorial! Keep practicing, explore advanced Java topics like multithreading, networking, and Java EE, and build real-world projects to solidify your Java programming skills.
            ''',
            'duration': 12  # Length of video in minutes
        },
        'fr': {
            'title': 'Bases de la programmation Java',
            'script': '''
                Introduction:
                    Bienvenue dans notre tutoriel de programmation Java ! Dans ce tutoriel, nous couvrirons les bases de Java, un langage de programmation largement utilisé pour la création d'applications, notamment dans le domaine du développement de logiciels d'entreprise.

                    1. Qu'est-ce que Java ?
                    Java est un langage de programmation orienté objet de haut niveau connu pour son indépendance de plateforme. Il vous permet d'écrire du code une seule fois et de l'exécuter sur n'importe quel appareil prenant en charge Java, le rendant ainsi très polyvalent.

                    2. Configuration de Java :
                    Pour commencer à coder en Java, vous devez installer le Kit de développement Java (JDK) sur votre ordinateur. Téléchargez le JDK depuis le site officiel et suivez les instructions d'installation en fonction de votre système d'exploitation.

                    3. Écrire votre premier programme :
                    Plongeons dans le codage avec Java ! Ouvrez un éditeur de texte ou un Environnement de Développement Intégré (IDE) comme IntelliJ IDEA ou Eclipse. Écrivez un programme simple pour afficher "Bonjour, monde !" à l'écran et exécutez-le pour voir le résultat.

                    4. Variables et types de données :
                    Java prend en charge différents types de données tels que les entiers, les nombres à virgule flottante, les caractères, les chaînes de caractères, les tableaux, et plus encore. Apprenez à déclarer des variables, effectuer des opérations arithmétiques et manipuler des chaînes en Java.

                    5. Instructions conditionnelles et boucles :
                    Utilisez des instructions conditionnelles telles que if, else et switch pour prendre des décisions dans vos programmes Java. Explorez des boucles comme for, while et do-while pour itérer sur des blocs de code de manière efficace.

                    6. Classes et objets :
                    Java est un langage orienté objet, ce qui signifie que vous travaillez avec des classes et des objets. Apprenez à définir des classes, créer des objets et accéder à leurs propriétés et méthodes.

                    7. Gestion des exceptions :
                    La gestion des exceptions est cruciale en programmation Java. Apprenez à utiliser les blocs try-catch, throw, throws et finally pour gérer les exceptions de manière élégante et maintenir la stabilité du programme.

                    8. Packages et bibliothèques :
                    Java propose un vaste écosystème de packages et de bibliothèques pour diverses fonctionnalités. Découvrez des bibliothèques populaires comme JavaFX pour le développement d'interfaces utilisateur graphiques (GUI), JDBC pour la connectivité aux bases de données et JUnit pour les tests unitaires.

                    9. Bonnes pratiques et conseils :
                    Suivez les bonnes pratiques comme les conventions de nommage, l'écriture de commentaires clairs et l'organisation de votre code en modules réutilisables pour améliorer la qualité et la maintenabilité du code.

                    10. Conclusion :
                    Félicitations pour avoir terminé notre tutoriel Java ! Continuez à pratiquer, explorez des sujets Java avancés comme le multithreading, le réseau et Java EE, et créez des projets concrets pour consolider vos compétences en programmation Java.
            ''',
            'duration': 14  # Length of video in minutes
        },
        'es': {
            'title': 'Conceptos básicos de programación en Java',
            'script': '''
                Introducción:
                    ¡Bienvenido a nuestro tutorial de programación en Java! En este tutorial, cubriremos los conceptos básicos de Java, un lenguaje de programación ampliamente utilizado para desarrollar aplicaciones, especialmente en el ámbito del desarrollo de software empresarial.

                    1. ¿Qué es Java?
                    Java es un lenguaje de programación de alto nivel y orientado a objetos conocido por su independencia de plataforma. Te permite escribir código una vez y ejecutarlo en cualquier dispositivo que admita Java, lo que lo hace altamente versátil.

                    2. Configuración de Java:
                    Para comenzar a programar en Java, necesitas instalar el Kit de Desarrollo de Java (JDK) en tu computadora. Descarga el JDK desde el sitio web oficial y sigue las instrucciones de instalación según tu sistema operativo.

                    3. Escribir tu primer programa:
                    ¡Vamos a sumergirnos en la codificación con Java! Abre un editor de texto o un Entorno de Desarrollo Integrado (IDE) como IntelliJ IDEA o Eclipse. Escribe un programa simple para mostrar "¡Hola, mundo!" en la pantalla y ejecútalo para ver la salida.

                    4. Variables y tipos de datos:
                    Java admite diferentes tipos de datos como enteros, números de punto flotante, caracteres, cadenas, matrices y más. Aprende a declarar variables, realizar operaciones aritméticas y manipular cadenas en Java.

                    5. Sentencias condicionales y bucles:
                    Utiliza sentencias condicionales como if, else y switch para tomar decisiones en tus programas Java. Explora bucles como for, while y do-while para iterar sobre bloques de código de manera eficiente.

                    6. Clases y objetos:
                    Java es un lenguaje orientado a objetos, lo que significa que trabajas con clases y objetos. Aprende a definir clases, crear objetos y acceder a sus propiedades y métodos.

                    7. Manejo de excepciones:
                    El manejo de excepciones es crucial en la programación Java. Aprende sobre bloques try-catch, throw, throws y finally para gestionar excepciones de manera elegante y mantener la estabilidad del programa.

                    8. Paquetes y bibliotecas:
                    Java ofrece un vasto ecosistema de paquetes y bibliotecas para diversas funcionalidades. Descubre bibliotecas populares como JavaFX para el desarrollo de interfaces gráficas de usuario (GUI), JDBC para la conectividad con bases de datos y JUnit para pruebas unitarias.

                    9. Buenas prácticas y consejos:
                    Sigue las mejores prácticas como convenciones de nomenclatura, escribir comentarios claros y organizar tu código en módulos reutilizables para mejorar la calidad y mantenibilidad del código.

                    10. Conclusión:
                    ¡Felicidades por completar nuestro tutorial de Java! Sigue practicando, explora temas avanzados de Java como multihilo, redes y Java EE, y crea proyectos del mundo real para consolidar tus habilidades de programación Java.
            ''',
            'duration': 15  # Length of video in minutes
        }
    },
    'cplusplus': {
        'en': {
            'title': 'C++ Programming Fundamentals',
            'script': '''
                Introduction:
                    Welcome to our C++ programming tutorial! In this tutorial, we'll cover the fundamentals of C++, a powerful and versatile programming language widely used for developing system software, games, and high-performance applications.

                    1. What is C++?
                    C++ is a middle-level, general-purpose programming language developed from C. It combines low-level features for system programming and high-level features for application development, making it suitable for various domains.

                    2. Setting Up C++:
                    To start coding in C++, you need a C++ compiler installed on your machine. Popular compilers include GCC, Clang, and Microsoft Visual C++. Choose one and set up your development environment.

                    3. Writing Your First Program:
                    Let's dive into coding with C++! Open a text editor or an Integrated Development Environment (IDE) like Visual Studio or Code::Blocks. Write a simple program to display "Hello, World!" and run it to see the output.

                    4. Variables and Data Types:
                    C++ supports various data types including integers, floating-point numbers, characters, strings, arrays, and more. Learn how to declare variables, perform arithmetic operations, and manipulate strings in C++.

                    5. Conditional Statements and Loops:
                    Use conditional statements like if, else, switch, and loops like for, while, and do-while to control the flow of your C++ programs based on conditions and repeat tasks efficiently.

                    6. Functions and Classes:
                    C++ is an object-oriented language supporting classes and functions. Learn how to define functions, create classes, work with constructors and destructors, and access class members.

                    7. Pointers and Memory Management:
                    Understanding pointers and memory management is crucial in C++. Learn about pointers, dynamic memory allocation, smart pointers, and best practices to avoid memory leaks and manage resources efficiently.

                    8. Standard Template Library (STL):
                    The STL provides powerful data structures and algorithms in C++. Explore containers like vectors, lists, maps, and algorithms for sorting, searching, and manipulating data efficiently.

                    9. Exception Handling:
                    Exception handling in C++ allows you to handle errors and unexpected situations gracefully. Learn about try-catch blocks, throw, catch, and exceptions to write robust and reliable C++ code.

                    10. Best Practices and Tips:
                    Follow best practices like naming conventions, writing modular and reusable code, using libraries effectively, and optimizing code for performance to become a proficient C++ programmer.

                    11. Conclusion:
                    Congratulations on completing our C++ tutorial! Keep practicing, explore advanced topics like templates, multithreading, and libraries like Boost, and work on projects to strengthen your C++ skills.
            ''',
            'duration': 15  # Length of video in minutes
        },
        'fr': {
            'title': 'Fondamentaux de la programmation C++',
            'script': '''
                Introduction:
                    Bienvenue dans notre tutoriel de programmation C++ ! Dans ce tutoriel, nous couvrirons les fondamentaux du C++, un langage de programmation puissant et polyvalent largement utilisé pour développer des logiciels système, des jeux et des applications haute performance.

                    1. Qu'est-ce que le C++ ?
                    Le C++ est un langage de programmation généraliste de niveau intermédiaire développé à partir de C. Il combine des fonctionnalités de bas niveau pour la programmation système et des fonctionnalités de haut niveau pour le développement d'applications, le rendant adapté à divers domaines.

                    2. Configuration du C++ :
                    Pour commencer à coder en C++, vous avez besoin d'un compilateur C++ installé sur votre machine. Les compilateurs populaires incluent GCC, Clang et Microsoft Visual C++. Choisissez-en un et configurez votre environnement de développement.

                    3. Écrire votre premier programme :
                    Plongeons dans le codage avec C++ ! Ouvrez un éditeur de texte ou un Environnement de Développement Intégré (IDE) comme Visual Studio ou Code::Blocks. Écrivez un programme simple pour afficher "Bonjour, monde !" et exécutez-le pour voir le résultat.

                    4. Variables et types de données :
                    Le C++ prend en charge différents types de données, notamment des entiers, des nombres à virgule flottante, des caractères, des chaînes de caractères, des tableaux, et plus encore. Apprenez à déclarer des variables, effectuer des opérations arithmétiques et manipuler des chaînes en C++.

                    5. Instructions conditionnelles et boucles :
                    Utilisez des instructions conditionnelles telles que if, else, switch et des boucles comme for, while, et do-while pour contrôler le flux de vos programmes C++ en fonction de conditions et répéter des tâches de manière efficace.

                    6. Fonctions et classes :
                    Le C++ est un langage orienté objet qui prend en charge les classes et les fonctions. Apprenez à définir des fonctions, créer des classes, travailler avec des constructeurs et destructeurs, et accéder aux membres de classe.

                    7. Pointeurs et gestion de la mémoire :
                    Comprendre les pointeurs et la gestion de la mémoire est crucial en C++. Apprenez les pointeurs, l'allocation dynamique de mémoire, les pointeurs intelligents et les meilleures pratiques pour éviter les fuites de mémoire et gérer efficacement les ressources.

                    8. Bibliothèque standard de modèles (STL) :
                    La STL offre des structures de données et des algorithmes puissants en C++. Explorez des conteneurs comme des vecteurs, des listes, des maps, et des algorithmes pour le tri, la recherche, et la manipulation de données de manière efficace.

                    9. Gestion des exceptions :
                    La gestion des exceptions en C++ vous permet de gérer les erreurs et les situations inattendues de manière élégante. Apprenez les blocs try-catch, throw, catch, et les exceptions pour écrire un code C++ robuste et fiable.

                    10. Bonnes pratiques et conseils :
                    Suivez les bonnes pratiques comme les conventions de nommage, l'écriture de code modulaire et réutilisable, l'utilisation efficace des bibliothèques, et l'optimisation du code pour les performances pour devenir un programmeur C++ compétent.

                    11. Conclusion :
                    Félicitations pour avoir terminé notre tutoriel C++ ! Continuez à pratiquer, explorez des sujets avancés comme les modèles, le multithreading, et les bibliothèques comme Boost, et travaillez sur des projets pour renforcer vos compétences en C++.
            ''',
            'duration': 18  # Length of video in minutes
        },
        'es': {
            'title': 'Conceptos fundamentales de programación en C++',
            'script': '''
                Introducción:
                    ¡Bienvenido a nuestro tutorial de programación en C++! En este tutorial, cubriremos los fundamentos de C++, un lenguaje de programación potente y versátil ampliamente utilizado para desarrollar software del sistema, juegos y aplicaciones de alto rendimiento.

                    1. ¿Qué es C++?
                    C++ es un lenguaje de programación general de nivel medio desarrollado a partir de C. Combina características de bajo nivel para la programación del sistema y características de alto nivel para el desarrollo de aplicaciones, lo que lo hace adecuado para diversos ámbitos.

                    2. Configuración de C++:
                    Para comenzar a programar en C++, necesitas un compilador C++ instalado en tu máquina. Los compiladores populares incluyen GCC, Clang y Microsoft Visual C++. Elige uno y configura tu entorno de desarrollo.

                    3. Escribir tu primer programa:
                    ¡Vamos a sumergirnos en la codificación con C++! Abre un editor de texto o un Entorno de Desarrollo Integrado (IDE) como Visual Studio o Code::Blocks. Escribe un programa simple para mostrar "¡Hola, mundo!" y ejecútalo para ver la salida.

                    4. Variables y tipos de datos:
                    C++ admite varios tipos de datos, incluidos enteros, números de punto flotante, caracteres, cadenas, matrices y más. Aprende a declarar variables, realizar operaciones aritméticas y manipular cadenas en C++.

                    5. Sentencias condicionales y bucles:
                    Utiliza sentencias condicionales como if, else, switch, y bucles como for, while, y do-while para controlar el flujo de tus programas C++ en función de condiciones y repetir tareas de manera eficiente.

                    6. Funciones y clases:
                    C++ es un lenguaje orientado a objetos que admite clases y funciones. Aprende a definir funciones, crear clases, trabajar con constructores y destructores, y acceder a miembros de clase.

                    7. Punteros y gestión de memoria:
                    Comprender los punteros y la gestión de memoria es crucial en C++. Aprende sobre punteros, asignación de memoria dinámica, punteros inteligentes y mejores prácticas para evitar pérdidas de memoria y gestionar recursos eficazmente.

                    8. Biblioteca Estándar de Plantillas (STL):
                    La STL ofrece estructuras de datos y algoritmos potentes en C++. Explora contenedores como vectores, listas, maps, y algoritmos para ordenar, buscar y manipular datos eficientemente.

                    9. Manejo de Excepciones:
                    El manejo de excepciones en C++ te permite manejar errores y situaciones inesperadas de manera elegante. Aprende sobre bloques try-catch, throw, catch y excepciones para escribir un código C++ robusto y confiable.

                    10. Buenas Prácticas y Consejos:
                    Sigue buenas prácticas como convenciones de nombres, escribir código modular y reutilizable, utilizar bibliotecas de manera efectiva y optimizar el código para obtener un rendimiento óptimo y ser un programador C++ competente.

                    11. Conclusión:
                    ¡Felicidades por completar nuestro tutorial de C++! Sigue practicando, explora temas avanzados como plantillas, multihilado y bibliotecas como Boost, y trabaja en proyectos para fortalecer tus habilidades en C++.
            ''',
            'duration': 18  # Length of video in minutes
        }
    }
}


    if creativity > 0.5:
        # Perform sentiment analysis using NLTK's Vader
        sentiment_scores = [sid.polarity_scores(paragraph)['compound'] for paragraph in paragraphs]
    # Get predefined data based on the subject and language
    topic_data = predefined_data.get(subject.lower())
    if topic_data:
        language_data = topic_data.get(language.lower())
        if language_data:
            title = language_data['title']
            script = language_data['script']

            # Split the script into paragraphs
            paragraphs = script.split("\n\n")

            # Combine paragraphs into the final script with separate paragraphs
            final_script = ""
            for paragraph in paragraphs:
                header = "<b>" + paragraph.splitlines()[0].strip() + ":</b>"  # Extract and format the header
                body = "<br>".join(paragraph.splitlines()[1:])  # Extract the body of the paragraph
                final_script += f"{header}\n{body}\n\n"

            return title, final_script, language_data['duration']
    return "Sorry, the specified subject or language is not available in the predefined data.", None, None


st.title('❤️ YouTube Script Writing Tool with TTS') 

# Captures User Inputs
generate_mode = st.radio("Choose Script Generation Mode:", ("Generate Automatically", "Create Manually"))

if generate_mode == "Generate Automatically": 
    # Automatically generate script based on predefined data
    prompt = st.text_input('Enter a topic (e.g., python, biology, math):')
    language_map = {'English': 'en', 'France': 'fr', 'Spanish': 'es', 'Germany': 'de', 'Italy': 'it'}
    language = st.selectbox('Select Language:', list(language_map.keys()))
    language_code = language_map.get(language)
    creativity = st.slider('Creativity Level ✨ (0 LOW || 1 HIGH)', 0.0, 1.0, 0.5, step=0.1)


    button_key = f"generate_script_{generate_mode}_{prompt}_{language_code}"

    # Create the "Generate Script" button 
    generate_script_button = st.button("Generate Script", key=button_key)

    # Event listener 
    if generate_script_button:
        title, script, duration = generate_script(prompt, language_code, creativity, )

        if script:
            # Display Title, Script, and Duration
            st.markdown(f'<p class="title-text">{title}</p>', unsafe_allow_html=True)
            st.markdown(script, unsafe_allow_html=True)
            st.subheader("Video Duration:⏳")
            st.write(f"{duration} minutes")

            # Remove HTML tags
            clean_script_content = BeautifulSoup(script, "html.parser").get_text()
            tts = gTTS(clean_script_content, lang=language_code)
            audio_file_path = f"{title}_auto.mp3"
            tts.save(audio_file_path)
            st.audio(audio_file_path, format='audio/mp3')  # Embed audio in the app
            st.success("TTS audio generated successfully!")

else:
    # Allow user to create the script manually with multiple paragraphs
    title = st.text_input('Please provide the title for the video:', key="title")
    num_paragraphs_manual = st.slider('Number of Paragraphs to Include:', 1, 5, 3)

    st.subheader("Manually Create Script:")
    script_output = ""
    for i in range(num_paragraphs_manual):
        paragraph_name = st.text_input(f'Enter Paragraph {i+1} Name:', key=f"para_name_{i}")
        paragraph_body = st.text_area(f'Enter Paragraph {i+1} Body:', key=f"para_body_{i}")
        if paragraph_name and paragraph_body:
            header = f"{paragraph_name}:"
            body = paragraph_body.replace('\n', '\n\n')  # Replace newlines with double newline for paragraphs
            script_output += f"{header}\n{body}\n\n"

    if title and script_output:
        # Display Title and Script
        st.markdown(f'<p class="title-text">{title}</p>', unsafe_allow_html=True)
        st.markdown(script_output, unsafe_allow_html=True)

        # Remove HTML tags from the script for TTS audio
        clean_script_content = BeautifulSoup(script_output, "html.parser").get_text()
        tts = gTTS(clean_script_content, lang='en')
        audio_file_path = f"{title}_auto.mp3"
        tts.save(audio_file_path)
        st.audio(audio_file_path, format='audio/mp3')  # Embed audio in the app
        st.success("TTS audio generated successfully!")

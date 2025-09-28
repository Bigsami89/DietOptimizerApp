package com.example.dietoptimizerapp.jni;

import java.util.ArrayList;
import com.example.dietoptimizerapp.models.DietResult;

public class JniBridge {
    static {
        try {
            System.loadLibrary("dietoptimizer");
        } catch (UnsatisfiedLinkError e) {
            e.printStackTrace();
            throw new RuntimeException("No se pudo cargar la librería nativa", e);
        }
    }

    public static native ArrayList<DietResult> calculateDiets(String jsonInput);
}
/**
 * The JPMS module for the JTransforms library.
 * Provides multithreaded FFT and other transforms in pure Java.
 */
module org.jtransforms {
    requires org.visnow.jlargearrays;
    requires java.logging;
    exports org.jtransforms.dct;
    exports org.jtransforms.dht;
    exports org.jtransforms.dst;
    exports org.jtransforms.fft;
}
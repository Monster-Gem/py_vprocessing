(asdf:defsystem #:cl-vprocessing
  :description "Video Processing Using Common Lisp"
  :author "Gustavo Alves Pacheco <gap1512@gmail.com>, João Barboza Rodrigues <joaobarboza.ufu@gmail.com>, Lucas Resende Carneiro <lucasrescarneiro@gmail.com>"
  :serial t
  :depends-on (#:pfft #:fft)
  :components ((:file "package")
	       (:file "./array-processing/array-processing")
	       (:file "./interface/interface")
	       (:file "./video-io/video-io")))
